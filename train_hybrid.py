import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common_metrics import calculate_metrics, save_confusion_matrix, save_results
from hybrid_utils import retrieval_probabilities
from project_utils import load_data, project_path

TRANSFORMER_PATH = project_path("models", "distilbert", "final")
SBERT_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
TEMPERATURE = 0.07
TRANSFORMER_WEIGHT = 0.75
MAX_LENGTH = 256
INFERENCE_BATCH_SIZE = 8


def transformer_probabilities(model, tokenizer, texts, device):
    model.eval()
    chunks = []

    for start in range(0, len(texts), INFERENCE_BATCH_SIZE):
        batch_texts = texts[start:start + INFERENCE_BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1)

        chunks.append(probabilities.cpu().numpy())

    return np.vstack(chunks)


def main():
    train_df, test_df = load_data()

    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(train_df["queue"])
    test_labels = encoder.transform(test_df["queue"])
    num_classes = len(encoder.classes_)

    if not TRANSFORMER_PATH.is_dir():
        raise FileNotFoundError(
            f"Could not find {TRANSFORMER_PATH}. "
            "Run `python train_transformers.py` first."
        )

    print("Loading the trained DistilBERT classifier...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_PATH))
    transformer = AutoModelForSequenceClassification.from_pretrained(
        str(TRANSFORMER_PATH)
    ).to(device)

    print("Predicting with DistilBERT...")
    transformer_prediction_start = time.time()
    transformer_probs = transformer_probabilities(
        transformer,
        tokenizer,
        test_df["text"].astype(str).tolist(),
        device,
    )
    transformer_prediction_time = time.time() - transformer_prediction_start

    print(f"Loading Sentence-BERT: {SBERT_NAME}")
    sbert = SentenceTransformer(SBERT_NAME)

    print("Encoding training tickets for retrieval...")
    embedding_start = time.time()
    train_embeddings = sbert.encode(
        train_df["text"].astype(str).tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("Encoding test tickets for retrieval...")
    test_embeddings = sbert.encode(
        test_df["text"].astype(str).tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    embedding_time = time.time() - embedding_start

    retrieval_start = time.time()
    retrieval_probs, top_indices = retrieval_probabilities(
        train_embeddings,
        test_embeddings,
        train_labels,
        num_classes,
        top_k=TOP_K,
        temperature=TEMPERATURE,
    )

    hybrid_probs = (
        TRANSFORMER_WEIGHT * transformer_probs
        + (1.0 - TRANSFORMER_WEIGHT) * retrieval_probs
    )

    transformer_predictions = transformer_probs.argmax(axis=1)
    hybrid_predictions = hybrid_probs.argmax(axis=1)
    retrieval_fusion_time = time.time() - retrieval_start
    prediction_time = transformer_prediction_time + retrieval_fusion_time

    transformer_metrics = calculate_metrics(
        test_labels,
        transformer_predictions,
        transformer_probs,
        num_classes,
    )
    hybrid_metrics = calculate_metrics(
        test_labels,
        hybrid_predictions,
        hybrid_probs,
        num_classes,
    )

    hit_at_3 = float(
        np.any(train_labels[top_indices] == test_labels[:, None], axis=1).mean()
    )

    try:
        with project_path("results", "distilbert_results.json").open(
            "r", encoding="utf-8"
        ) as file:
            distilbert_info = json.load(file)
        distilbert_training_time = distilbert_info.get("training_time_seconds")
    except FileNotFoundError:
        distilbert_training_time = None

    result = {
        **hybrid_metrics,
        "model": "Hybrid DistilBERT + SBERT probability fusion",
        "base_classifier": "distilbert-base-multilingual-cased",
        "embedding_model": SBERT_NAME,
        "transformer_weight": TRANSFORMER_WEIGHT,
        "retrieval_weight": 1.0 - TRANSFORMER_WEIGHT,
        "retrieval_top_k": TOP_K,
        "retrieval_temperature": TEMPERATURE,
        "retrieval_hit_at_3": hit_at_3,
        "training_time_seconds": distilbert_training_time,
        "embedding_time_seconds": round(embedding_time, 4),
        "prediction_time_seconds": round(prediction_time, 4),
        "transformer_prediction_time_seconds": round(transformer_prediction_time, 4),
        "retrieval_fusion_time_seconds": round(retrieval_fusion_time, 4),
        "transformer_baseline": transformer_metrics,
        "hybrid_metrics": hybrid_metrics,
        "note": (
            "The hybrid reuses the trained DistilBERT model and adds a semantic "
            "retrieval probability distribution. Training time is DistilBERT's "
            "training time; embedding time is reported separately."
        ),
    }

    model_dir = project_path("models", "hybrid")
    model_dir.mkdir(parents=True, exist_ok=True)
    sbert.save(str(model_dir / "sbert"))
    joblib.dump(encoder, model_dir / "label_encoder.joblib")
    np.save(model_dir / "historical_embeddings.npy", train_embeddings)
    train_df[
        [col for col in ("text", "queue", "priority", "type") if col in train_df.columns]
    ].to_csv(model_dir / "historical_tickets.csv", index=False)

    save_results(project_path("results", "hybrid_results.json"), result)
    save_confusion_matrix(
        project_path("results", "hybrid_confusion_matrix.csv"),
        test_labels,
        hybrid_predictions,
    )
    save_confusion_matrix(
        project_path("results", "hybrid_transformer_confusion_matrix.csv"),
        test_labels,
        transformer_predictions,
    )

    report = (
        "TRANSFORMER-ONLY BASELINE\n"
        + json.dumps(transformer_metrics, indent=2)
        + "\n\nHYBRID RESULTS\n"
        + json.dumps(hybrid_metrics, indent=2)
        + f"\n\nRetrieval Hit@3: {hit_at_3:.6f}\n"
    )
    with project_path("results", "hybrid_report.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write(report)

    print("\nCOMPARISON ON THE SAME TEST SET")
    for name, metrics in (("DistilBERT", transformer_metrics), ("Hybrid", hybrid_metrics)):
        print(f"\n{name}")
        print(f"Accuracy:       {metrics['accuracy']:.4f}")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1:             {metrics['f1']:.4f}")
        print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    print(f"\nRetrieval Hit@3: {hit_at_3:.4f}")
    print(f"Embedding time: {embedding_time:.2f} seconds")
    print(f"Saved: {project_path('results', 'hybrid_results.json')}")


if __name__ == "__main__":
    main()

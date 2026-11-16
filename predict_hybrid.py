import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hybrid_utils import retrieval_probabilities
from project_utils import project_path

MODEL_DIR = project_path("models", "hybrid")
TRANSFORMER_DIR = project_path("models", "distilbert", "final")
SBERT_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRANSFORMER_WEIGHT = 0.75
TOP_K = 3
TEMPERATURE = 0.07
MAX_LENGTH = 256


def main():
    required_files = [
        TRANSFORMER_DIR,
        MODEL_DIR / "sbert",
        MODEL_DIR / "label_encoder.joblib",
        MODEL_DIR / "historical_embeddings.npy",
        MODEL_DIR / "historical_tickets.csv",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The hybrid model is not ready. Run `python train_hybrid.py` first.\n"
            + "\n".join(missing)
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_DIR))
    transformer = AutoModelForSequenceClassification.from_pretrained(
        str(TRANSFORMER_DIR)
    ).to(device)
    transformer.eval()

    sbert = SentenceTransformer(str(MODEL_DIR / "sbert"))
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    train_embeddings = np.load(MODEL_DIR / "historical_embeddings.npy").astype(np.float32)
    history = pd.read_csv(MODEL_DIR / "historical_tickets.csv", low_memory=False)
    train_labels = encoder.transform(history["queue"].astype(str))

    subject = input("Enter ticket subject: ").strip()
    body = input("Enter ticket description: ").strip()
    text = f"{subject} {body}".strip()
    if not text:
        raise ValueError("Enter a subject or a description.")

    tokenized = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    with torch.no_grad():
        transformer_probs = torch.softmax(transformer(**tokenized).logits, dim=1).cpu().numpy()

    query_embedding = sbert.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    retrieval_probs, top_indices = retrieval_probabilities(
        train_embeddings,
        query_embedding,
        train_labels,
        len(encoder.classes_),
        top_k=TOP_K,
        temperature=TEMPERATURE,
    )

    fused_probs = (
        TRANSFORMER_WEIGHT * transformer_probs
        + (1.0 - TRANSFORMER_WEIGHT) * retrieval_probs
    )[0]

    predicted_index = int(fused_probs.argmax())
    predicted_queue = encoder.inverse_transform([predicted_index])[0]

    print("\n================================")
    print("HYBRID TICKET TRIAGE RESULT")
    print("================================")
    print("Predicted queue:", predicted_queue)
    print(
        "Hybrid model score:",
        f"{float(fused_probs[predicted_index]) * 100:.2f}%",
        "(not probability-calibrated)",
    )

    print("\nSimilar historical tickets:")
    for rank, index in enumerate(top_indices[0], start=1):
        row = history.iloc[int(index)]
        similarity = float(query_embedding[0] @ train_embeddings[int(index)])
        print("\n----------------------------")
        print("Rank:", rank)
        print("Queue:", row.get("queue", "Not available"))
        print("Priority:", row.get("priority", "Not available"))
        print("Type:", row.get("type", "Not available"))
        print(f"Cosine similarity: {similarity:.4f}")
        print("Text:", str(row.get("text", ""))[:300])


if __name__ == "__main__":
    main()

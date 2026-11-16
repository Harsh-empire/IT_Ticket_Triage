import time

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from common_metrics import calculate_metrics, save_confusion_matrix, save_results
from project_utils import load_data, project_path

SBERT_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_STATE = 42


def main():
    train_df, test_df = load_data()

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df["queue"])
    y_test = encoder.transform(test_df["queue"])

    print(f"Loading sentence embedding model: {SBERT_NAME}")
    embedder = SentenceTransformer(SBERT_NAME)

    print("Encoding training tickets...")
    embedding_start = time.time()
    x_train = embedder.encode(
        train_df["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("Encoding testing tickets...")
    x_test = embedder.encode(
        test_df["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    embedding_time = time.time() - embedding_start

    print("\nTraining XGBoost on Sentence-BERT embeddings...")
    start = time.time()
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(x_train, y_train)
    training_time = time.time() - start

    start = time.time()
    probabilities = model.predict_proba(x_test)
    predictions = probabilities.argmax(axis=1)
    prediction_time = time.time() - start

    metrics = calculate_metrics(y_test, predictions, probabilities, len(encoder.classes_))
    metrics.update({
        "model": "Sentence-BERT + XGBoost",
        "embedding_model": SBERT_NAME,
        "training_time_seconds": round(training_time, 4),
        "embedding_time_seconds": round(embedding_time, 4),
        "prediction_time_seconds": round(prediction_time, 4),
    })

    model_dir = project_path("models", "sbert_xgboost")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(encoder, model_dir / "label_encoder.joblib")
    embedder.save(str(model_dir / "embedding_model"))

    results_path = project_path("results", "sbert_xgboost_results.json")
    save_results(results_path, metrics)
    save_confusion_matrix(
        project_path("results", "sbert_xgboost_confusion_matrix.csv"),
        y_test,
        predictions,
    )

    print("\nSentence-BERT + XGBoost results")
    for key in ("accuracy", "precision", "recall", "f1", "top_3_accuracy"):
        print(f"{key}: {metrics.get(key)}")
    print(f"Embedding time: {embedding_time:.2f} seconds")
    print(f"Classifier training time: {training_time:.2f} seconds")
    print(f"Prediction time: {prediction_time:.2f} seconds")


if __name__ == "__main__":
    main()

import json

import pandas as pd

from project_utils import project_path

RESULT_FILES = [
    ("TF-IDF + Logistic Regression", "tfidf_logistic_regression_results.json"),
    ("TF-IDF + XGBoost", "tfidf_xgboost_results.json"),
    ("Sentence-BERT + XGBoost", "sbert_xgboost_results.json"),
    ("DistilBERT", "distilbert_results.json"),
    ("BERT", "bert_results.json"),
    ("Hybrid DistilBERT + SBERT", "hybrid_results.json"),
]


def main():
    rows = []

    for display_name, filename in RESULT_FILES:
        path = project_path("results", filename)
        if not path.is_file():
            print(f"Not available yet, skipping: {path.name}")
            continue

        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)

        rows.append({
            "Model": display_name,
            "Accuracy": result.get("accuracy"),
            "Precision": result.get("precision"),
            "Recall": result.get("recall"),
            "F1": result.get("f1"),
            "Top-3 Accuracy": result.get("top_3_accuracy"),
            "Training Time (seconds)": result.get("training_time_seconds"),
            "Prediction Time (seconds)": result.get("prediction_time_seconds"),
            "Embedding Time (seconds)": result.get("embedding_time_seconds"),
            "Retrieval Hit@3": result.get("retrieval_hit_at_3"),
        })

    if not rows:
        raise FileNotFoundError(
            "No model results found. Run the training scripts first."
        )

    table = pd.DataFrame(rows)
    output_path = project_path("results", "model_comparison.csv")
    table.to_csv(output_path, index=False)

    print("\nMODEL COMPARISON")
    print("F1 is weighted F1. Dashes mean that metric does not apply or was not recorded.\n")
    print(table.fillna("—").to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

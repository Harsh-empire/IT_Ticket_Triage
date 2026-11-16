import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from common_metrics import calculate_metrics, save_confusion_matrix, save_results
from project_utils import load_data, project_path

RANDOM_STATE = 42


def evaluate_and_save(name, model, x_test, y_test, encoder, training_time, vectorizer=None):
    start = time.time()
    probabilities = model.predict_proba(x_test)
    predictions = probabilities.argmax(axis=1)
    prediction_time = time.time() - start

    metrics = calculate_metrics(y_test, predictions, probabilities, len(encoder.classes_))
    metrics.update({
        "model": name,
        "training_time_seconds": round(training_time, 4),
        "prediction_time_seconds": round(prediction_time, 4),
        "embedding_time_seconds": None,
    })

    model_dir = project_path("models", name.lower().replace(" ", "_").replace("+", "plus"))
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(encoder, model_dir / "label_encoder.joblib")
    if vectorizer is not None:
        joblib.dump(vectorizer, model_dir / "tfidf.joblib")

    result_names = {
        "TF-IDF + Logistic Regression": "tfidf_logistic_regression",
        "TF-IDF + XGBoost": "tfidf_xgboost",
    }
    result_stem = result_names[name]
    result_file = project_path("results", f"{result_stem}_results.json")
    save_results(result_file, metrics)
    save_confusion_matrix(
        project_path("results", f"{result_stem}_confusion_matrix.csv"),
        y_test,
        predictions,
    )

    print(f"\n{name}")
    for key in ("accuracy", "precision", "recall", "f1", "top_3_accuracy"):
        print(f"{key}: {metrics.get(key)}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Prediction time: {prediction_time:.2f} seconds")
    return metrics


def main():
    train_df, test_df = load_data()
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df["queue"])
    y_test = encoder.transform(test_df["queue"])

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words=None,
        dtype=np.float32,
    )

    print("Building TF-IDF features...")
    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(test_df["text"])

    print("\nTraining Logistic Regression...")
    start = time.time()
    logistic = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    logistic.fit(x_train, y_train)
    logistic_time = time.time() - start

    # Save one shared encoder/vectorizer with the corresponding model folders.
    evaluate_and_save(
        "TF-IDF + Logistic Regression",
        logistic,
        x_test,
        y_test,
        encoder,
        logistic_time,
        vectorizer,
    )

    print("\nTraining TF-IDF + XGBoost...")
    start = time.time()
    xgb_model = XGBClassifier(
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
    xgb_model.fit(x_train, y_train)
    xgb_time = time.time() - start

    evaluate_and_save(
        "TF-IDF + XGBoost",
        xgb_model,
        x_test,
        y_test,
        encoder,
        xgb_time,
        vectorizer,
    )


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_pred, probabilities, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        # The main comparison table has one F1 column, defined as weighted F1.
        "f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "top_3_accuracy": None,
    }

    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        k = min(3, num_classes, probabilities.shape[1])
        if k > 0:
            top_indices = np.argpartition(
                probabilities,
                kth=probabilities.shape[1] - k,
                axis=1,
            )[:, -k:]
            result["top_3_accuracy"] = float(
                np.any(top_indices == y_true[:, None], axis=1).mean()
            )

        try:
            result["roc_auc_ovr_weighted"] = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    labels=np.arange(num_classes),
                    multi_class="ovr",
                    average="weighted",
                )
            )
        except (ValueError, TypeError):
            # ROC AUC may be undefined when a class is missing from the test set.
            pass

    return result


def save_results(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


def save_confusion_matrix(path, y_true, y_pred):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    np.savetxt(path, matrix, delimiter=",", fmt="%d")

"""Evaluation helpers for held-out model predictions."""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.prediction import LABELS


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(
            y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
        "labels": list(LABELS),
    }

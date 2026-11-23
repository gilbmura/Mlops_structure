"""Evaluation metrics for the four-class model and its derived tumor / no-tumor status.

Beyond accuracy and per-class precision/recall/F1, three extra measures are reported:

* balanced accuracy and macro ROC-AUC (one-vs-rest): robust to class imbalance and,
  for AUC, independent of any decision threshold;
* tumor-detection sensitivity/specificity: the UI's headline message is derived
  from the four-class output, so its binary error rates are measured directly;
* expected calibration error (ECE): the UI shows a confidence score, so we check
  how far that score is from observed accuracy.
"""

from __future__ import annotations

import keras
import numpy as np
from sklearn import metrics

from src.labels import CLASS_NAMES, CLASS_TO_INDEX, NO_TUMOR
from src.preprocessing import to_model_input


def predict_probabilities(model: keras.Model, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Softmax probabilities for standardized ``uint8`` images ``(N, H, W)``."""
    return model.predict(to_model_input(images), batch_size=batch_size, verbose=0)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    """Weighted mean gap between confidence and accuracy over equal-width confidence bins."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def _tumor_detection(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary view: any tumor class counts as positive."""
    no_tumor = CLASS_TO_INDEX[NO_TUMOR]
    actual, predicted = y_true != no_tumor, y_pred != no_tumor
    tn, fp, fn, tp = metrics.confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    return {
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "accuracy": float((tp + tn) / len(y_true)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def classification_report(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """All headline metrics as JSON-serialisable values."""
    labels = list(range(len(CLASS_NAMES)))
    y_pred = probabilities.argmax(axis=1)
    precision, recall, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = np.unique(y_true)
    report = {
        "samples": int(len(y_true)),
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(metrics.f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": metrics.confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_order": list(CLASS_NAMES),
        "log_loss": float(metrics.log_loss(y_true, probabilities, labels=labels)),
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "tumor_detection": _tumor_detection(y_true, y_pred),
    }
    if len(present) == len(CLASS_NAMES):
        report["macro_roc_auc_ovr"] = float(metrics.roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
    return report


def evaluate_model(model: keras.Model, images: np.ndarray, labels: np.ndarray) -> tuple[dict, np.ndarray]:
    """Predict and score; returns the report and the raw probabilities."""
    probabilities = predict_probabilities(model, images)
    return classification_report(labels, probabilities), probabilities

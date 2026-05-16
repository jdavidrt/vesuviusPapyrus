"""Binary classification metrics implemented from scratch."""

from __future__ import annotations

import numpy as np


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN). Both arrays must be binary."""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true {y_true.shape} and y_pred {y_pred.shape} must match."
        )
    yt = y_true.astype(np.int64).ravel()
    yp = y_pred.astype(np.int64).ravel()
    if not np.all((yt == 0) | (yt == 1)):
        raise ValueError("y_true must contain only 0 and 1.")
    if not np.all((yp == 0) | (yp == 1)):
        raise ValueError("y_pred must contain only 0 and 1.")
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    return tp, fp, fn, tn


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return accuracy, precision, recall, F1 for a binary classification."""
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
    total = tp + fp + fn + tn
    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

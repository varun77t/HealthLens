"""Classification metrics for the disease models.

Everything is computed from ``(y_true, y_prob)`` where ``y_prob`` is the model's
estimated probability of the positive class. The default decision threshold is 0.5
but a sensitivity-oriented alternative is always reported alongside, because for a
screening-style model recall usually matters more than raw accuracy.

No metric here is a placeholder — each is a direct scikit-learn computation on
arrays the caller supplies from an actual model run.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def confusion_at(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Full metric suite at a given probability threshold.

    Threshold-independent metrics (ROC-AUC, PR-AUC, Brier) are also included.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": _specificity(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    out.update({f"cm_{k}": v for k, v in confusion_at(y_true, y_prob, threshold).items()})
    return out


def threshold_sweep(y_true, y_prob, thresholds=None):
    """DataFrame of the confusion-derived metrics across a grid of thresholds."""
    import pandas as pd

    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for t in thresholds:
        m = classification_metrics(y_true, y_prob, threshold=float(t))
        rows.append(
            {
                "threshold": float(t),
                "precision": m["precision"],
                "recall_sensitivity": m["recall_sensitivity"],
                "specificity": m["specificity"],
                "f1": m["f1"],
                "accuracy": m["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def youden_threshold(y_true, y_prob) -> float:
    """Threshold maximising Youden's J (sensitivity + specificity - 1).

    A common sensitivity-oriented operating point; reported, not imposed.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float))
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])

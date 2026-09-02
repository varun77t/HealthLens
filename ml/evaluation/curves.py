"""Evaluation plots: ROC, precision-recall, confusion matrix, calibration curve.

Each function saves a PNG and returns its path (same convention as ``ml.eda.plots``).
All take ``(y_true, y_prob)`` from an actual model run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sns.set_theme(style="whitegrid", palette="muted")


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc(y_true, y_prob, out_dir: Path, *, title: str = "") -> Path:
    y_true = np.asarray(y_true).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="#4C72B0", label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve {title}".strip())
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out_dir, "roc_curve.png")


def plot_precision_recall(y_true, y_prob, out_dir: Path, *, title: str = "") -> Path:
    y_true = np.asarray(y_true).astype(int)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(rec, prec, color="#C44E52", label=f"PR (AP = {ap:.3f})")
    ax.axhline(y_true.mean(), ls="--", color="grey", linewidth=1, label=f"prevalence = {y_true.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve {title}".strip())
    ax.legend(loc="best", fontsize=8)
    return _save(fig, out_dir, "pr_curve.png")


def plot_confusion_matrix(y_true, y_prob, out_dir: Path, *, threshold: float = 0.5, title: str = "") -> Path:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix (t = {threshold:g}) {title}".strip())
    return _save(fig, out_dir, "confusion_matrix.png")


def plot_calibration_curve(y_true, y_prob, out_dir: Path, *, n_bins: int = 10, title: str = "") -> Path:
    y_true = np.asarray(y_true).astype(int)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(mean_pred, frac_pos, "o-", color="#4C72B0", label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration {title}".strip())
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out_dir, "calibration_curve.png")


def all_evaluation_plots(y_true, y_prob, out_dir: Path, *, threshold: float = 0.5, title: str = "") -> list[Path]:
    return [
        plot_roc(y_true, y_prob, out_dir, title=title),
        plot_precision_recall(y_true, y_prob, out_dir, title=title),
        plot_confusion_matrix(y_true, y_prob, out_dir, threshold=threshold, title=title),
        plot_calibration_curve(y_true, y_prob, out_dir, title=title),
    ]

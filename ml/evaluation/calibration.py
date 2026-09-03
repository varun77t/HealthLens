"""Probability-calibration comparison and selection.

A model can rank cases well (high ROC-AUC) yet output probabilities that are
systematically off. For a risk-communication tool the probability itself matters, so
before freezing a model we compare three options:

    * ``raw``      - the pipeline's own ``predict_proba``
    * ``sigmoid``  - ``CalibratedClassifierCV(method="sigmoid")``  (Platt scaling)
    * ``isotonic`` - ``CalibratedClassifierCV(method="isotonic")``

**Selection happens on the training set only** (:func:`calibration_cv_scores` uses
``cross_val_predict``, so every probability scored is an out-of-fold prediction).
Choosing the method by test-set Brier would be a form of test-set selection and would
make the reported test performance optimistic. :func:`compare_calibration` still
computes the test-set numbers for all three, but purely for transparent reporting —
never feed its output to :func:`pick_calibration`.

The calibrators themselves are always cross-fitted inside ``CalibratedClassifierCV``,
so a calibrator is never fitted on the same rows as the model it calibrates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict

from config import CV_FOLDS, RANDOM_STATE

METHODS = ("raw", "sigmoid", "isotonic")


@dataclass
class CalibrationChoice:
    method: str
    selection_table: pd.DataFrame  # training-set CV scores that drove the choice
    rationale: str


def _score(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def _wrap(base_pipeline, method: str, inner_cv: int):
    if method == "raw":
        return clone(base_pipeline)
    return CalibratedClassifierCV(clone(base_pipeline), method=method, cv=inner_cv)


def calibration_cv_scores(
    base_pipeline,
    X_train,
    y_train,
    *,
    cv=None,
    groups=None,
    inner_cv: int = 3,
    methods=METHODS,
) -> pd.DataFrame:
    """Out-of-fold Brier / ROC-AUC on the **training** set for each calibration option.

    This is the table that decides which wrapper is used. Nested cross-validation:
    the outer folds (``cv``) produce the out-of-fold probabilities; the inner
    ``CalibratedClassifierCV`` folds (``inner_cv``) fit the calibrator.
    """
    from sklearn.model_selection import StratifiedKFold

    cv = cv or StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for method in methods:
        est = _wrap(base_pipeline, method, inner_cv)
        proba = cross_val_predict(
            est, X_train, y_train, cv=cv, groups=groups, method="predict_proba", n_jobs=-1
        )[:, 1]
        rows.append({"method": method, **_score(y_train, proba)})
    return pd.DataFrame(rows).set_index("method")


def pick_calibration(cv_table: pd.DataFrame, *, auc_tol: float = 0.01) -> CalibrationChoice:
    """Lowest out-of-fold Brier among options within ``auc_tol`` of raw ROC-AUC."""
    raw_auc = float(cv_table.loc["raw", "roc_auc"])
    eligible = cv_table[cv_table["roc_auc"] >= raw_auc - auc_tol]
    best = str(eligible["brier"].idxmin())
    rationale = (
        f"Selected '{best}' by out-of-fold Brier score on the training set: "
        f"{cv_table.loc[best, 'brier']:.4f} vs raw {cv_table.loc['raw', 'brier']:.4f}; "
        f"cross-validated ROC-AUC {cv_table.loc[best, 'roc_auc']:.4f} vs raw {raw_auc:.4f} "
        f"(tolerance {auc_tol}). The test set played no part in this choice."
    )
    return CalibrationChoice(method=best, selection_table=cv_table, rationale=rationale)


def fit_calibrated(base_pipeline, X_train, y_train, method: str, *, inner_cv: int = 3):
    """Fit the chosen wrapper on the full training set and return it."""
    est = _wrap(base_pipeline, method, inner_cv)
    est.fit(X_train, y_train)
    return est


def compare_calibration(
    base_pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    inner_cv: int = 3,
    methods=METHODS,
) -> pd.DataFrame:
    """Test-set Brier / ROC-AUC for each option — **reporting only, not selection**."""
    rows = []
    for method in methods:
        est = _wrap(base_pipeline, method, inner_cv)
        est.fit(X_train, y_train)
        rows.append({"method": method, **_score(y_test, est.predict_proba(X_test)[:, 1])})
    return pd.DataFrame(rows).set_index("method")


# --------------------------------------------------------------------------------------
# Phase 7 — calibration reporting (does not participate in selection)
# --------------------------------------------------------------------------------------


def expected_calibration_error(y_true, y_prob, *, n_bins: int = 10) -> float:
    """Expected Calibration Error: mean |confidence - accuracy| across probability bins.

    Brier score mixes calibration with discrimination — a model can improve its Brier by
    separating the classes better without its probabilities becoming any more truthful.
    ECE isolates the calibration part: within each bin of predicted probability, how far
    is the average prediction from the observed frequency. Bins are weighted by their
    population, and empty bins contribute nothing.

    Reported alongside Brier, never instead of it: ECE is blind to ranking, so a model
    that predicts the base rate for everybody scores a near-perfect ECE while being
    useless. The two numbers only mean something together.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(total)


def calibration_summary(y_true, p_raw, p_calibrated, *, method: str, n_bins: int = 10) -> dict:
    """Before/after calibration quality on the same rows.

    ``method == "raw"`` means no wrapper was applied, so the two columns are identical by
    construction and the deltas are zero — that is a legitimate outcome (the base model
    was already the best-calibrated option on training out-of-fold Brier), not a bug.
    """
    pre = {
        "brier": float(brier_score_loss(y_true, p_raw)),
        "ece": expected_calibration_error(y_true, p_raw, n_bins=n_bins),
        "roc_auc": float(roc_auc_score(y_true, p_raw)),
    }
    post = {
        "brier": float(brier_score_loss(y_true, p_calibrated)),
        "ece": expected_calibration_error(y_true, p_calibrated, n_bins=n_bins),
        "roc_auc": float(roc_auc_score(y_true, p_calibrated)),
    }
    return {
        "method_applied": method,
        "wrapper_applied": method != "raw",
        "n": int(len(y_true)),
        "n_bins": n_bins,
        "before": {k: round(v, 6) for k, v in pre.items()},
        "after": {k: round(v, 6) for k, v in post.items()},
        "delta_brier": round(post["brier"] - pre["brier"], 6),
        "delta_ece": round(post["ece"] - pre["ece"], 6),
        "delta_roc_auc": round(post["roc_auc"] - pre["roc_auc"], 6),
        "note": (
            "Brier and ECE are computed on the held-out test set for reporting. The "
            "calibration method itself was chosen on training out-of-fold Brier alone; "
            "these figures played no part in that choice."
        ),
    }


def plot_reliability_comparison(y_true, series: dict, out_dir, *, n_bins: int = 10,
                                title: str = "") -> "Path":
    """Overlay reliability curves for several probability columns on one axis.

    A single calibration curve shows whether a model is calibrated. Two overlaid show
    what the calibration wrapper actually did, which is the question this phase asks.
    """
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    for name, probs in series.items():
        frac, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac, marker="o", linewidth=1.5, label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Reliability — before vs after calibration {title}".strip())
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "calibration_before_after.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

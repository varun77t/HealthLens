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

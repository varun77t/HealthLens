"""Probability-calibration comparison.

A model can rank cases well (high ROC-AUC) yet output probabilities that are
systematically off. For a risk-communication tool the probability itself matters,
so before freezing a model we compare:

    * raw            - the fitted pipeline's own ``predict_proba``
    * sigmoid (Platt) - ``CalibratedClassifierCV(method="sigmoid")``
    * isotonic        - ``CalibratedClassifierCV(method="isotonic")``

on Brier score (lower = better) and ROC-AUC (must not drop). ``pick_calibration``
returns the option with the best Brier that does not lose more than ``auc_tol``
ROC-AUC versus raw.

The calibrators are always cross-fitted on the **training** data only
(``CalibratedClassifierCV(cv=...)``); the held-out test set is used purely to
score the three options.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

from config import CV_FOLDS, RANDOM_STATE

METHODS = ("raw", "sigmoid", "isotonic")


@dataclass
class CalibrationChoice:
    method: str
    table: pd.DataFrame
    rationale: str


def _score(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def compare_calibration(
    fitted_pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    cv: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Score raw / sigmoid / isotonic probabilities on the test set.

    ``fitted_pipeline`` must already be fitted on ``(X_train, y_train)``; the
    calibrated variants are refit from an unfitted clone via internal CV on the
    training data.
    """
    rows = [{"method": "raw", **_score(y_test, fitted_pipeline.predict_proba(X_test)[:, 1])}]
    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(clone(fitted_pipeline), method=method, cv=cv)
        calibrated.fit(X_train, y_train)
        rows.append({"method": method, **_score(y_test, calibrated.predict_proba(X_test)[:, 1])})
    return pd.DataFrame(rows).set_index("method")


def pick_calibration(table: pd.DataFrame, *, auc_tol: float = 0.01) -> CalibrationChoice:
    """Choose the lowest-Brier method that stays within ``auc_tol`` of raw ROC-AUC."""
    raw_auc = float(table.loc["raw", "roc_auc"])
    eligible = table[table["roc_auc"] >= raw_auc - auc_tol]
    best = eligible["brier"].idxmin()
    rationale = (
        f"Selected '{best}': Brier {table.loc[best, 'brier']:.4f} "
        f"(raw {table.loc['raw', 'brier']:.4f}), ROC-AUC {table.loc[best, 'roc_auc']:.4f} "
        f"(raw {raw_auc:.4f}, tolerance {auc_tol})."
    )
    return CalibrationChoice(method=str(best), table=table, rationale=rationale)

"""Diagnostics that test *why* a model scores well, not just how well.

A high ROC-AUC is not evidence that a model learned clinical structure. It can also
come from an artefact of how the data was collected. These probes try to attribute the
performance, and they are meant to be run and reported even — especially — when the
answer is unflattering.

``missingness_only_probe`` is the sharpest of them: it throws away every measured value
and trains only on *which values are missing*. If that alone reproduces most of the
headline score, then the model is substantially reading the measurement pattern (which
labs a clinician chose to order) rather than the measurements. That is not train/test
leakage — the split stays clean — but it is a hard ceiling on how far the result
transfers to a setting where the tests are ordered routinely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from ml.evaluation.metrics import classification_metrics


def _indicator_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Binary 'this cell was missing' matrix, keeping only columns that vary."""
    ind = X.isna().astype(int)
    return ind.loc[:, ind.nunique() > 1]


def missingness_only_probe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    cv=None,
    groups=None,
    random_state: int = RANDOM_STATE,
) -> dict | None:
    """Train on missingness indicators alone. Returns ``None`` if nothing is missing.

    The comparison of interest is this probe's ROC-AUC against the real model's. Close
    together means the real model's advantage over "which labs were ordered" is small.
    """
    ind_train = _indicator_frame(X_train)
    if ind_train.empty:
        return None
    ind_test = X_test.isna().astype(int).reindex(columns=ind_train.columns, fill_value=0)

    model = Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=random_state)),
        ]
    )
    result: dict = {
        "n_indicator_features": int(ind_train.shape[1]),
        "indicator_columns": list(ind_train.columns),
    }
    if cv is not None:
        scores = cross_val_score(
            model, ind_train, y_train, groups=groups, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        result["cv_roc_auc_mean"] = float(np.mean(scores))
        result["cv_roc_auc_std"] = float(np.std(scores))

    model.fit(ind_train, y_train)
    y_prob = model.predict_proba(ind_test)[:, 1]
    metrics = classification_metrics(y_test, y_prob, threshold=0.5)
    result["test_roc_auc"] = metrics["roc_auc"]
    result["test_pr_auc"] = metrics["pr_auc"]
    result["test_accuracy"] = metrics["accuracy"]
    return result


def complete_case_probe(X: pd.DataFrame, y: pd.Series) -> dict:
    """How much data survives if rows with any missing value are dropped?

    Reported to show why complete-case analysis was not used as the alternative.
    """
    keep = ~X.isna().any(axis=1)
    y = pd.Series(y).reset_index(drop=True)
    kept = y[keep.reset_index(drop=True)]
    return {
        "n_total": int(len(X)),
        "n_complete_cases": int(keep.sum()),
        "pct_retained": round(100 * float(keep.mean()), 2),
        "positive_rate_all": round(float(y.mean()), 4),
        "positive_rate_complete_cases": round(float(kept.mean()), 4) if len(kept) else float("nan"),
    }


def attribution_note(probe: dict | None, model_test_roc_auc: float) -> str:
    """One honest sentence comparing the probe against the real model."""
    if not probe:
        return "No values are missing in this dataset, so measurement pattern cannot carry signal."
    probe_auc = probe["test_roc_auc"]
    gap = model_test_roc_auc - probe_auc
    if probe_auc >= 0.90:
        strength = (
            "Missingness indicators ALONE — with every measured value discarded — reach "
            f"test ROC-AUC {probe_auc:.4f}, against the full model's {model_test_roc_auc:.4f} "
            f"(gap {gap:+.4f}). The measurement pattern, not the measurements, carries most "
            "of the separability. Treat the headline score as a property of how this dataset "
            "was collected, not as evidence of clinical usefulness."
        )
    elif probe_auc >= 0.70:
        strength = (
            f"Missingness indicators alone reach test ROC-AUC {probe_auc:.4f} against the full "
            f"model's {model_test_roc_auc:.4f} (gap {gap:+.4f}). A meaningful part of the "
            "signal comes from which values were recorded rather than their values."
        )
    else:
        strength = (
            f"Missingness indicators alone reach only test ROC-AUC {probe_auc:.4f} against the "
            f"full model's {model_test_roc_auc:.4f} (gap {gap:+.4f}), so the measurement "
            "pattern does not explain most of the performance."
        )
    return strength

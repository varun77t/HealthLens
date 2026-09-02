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


def duplicate_label_conflict_probe(X: pd.DataFrame, y: pd.Series) -> dict:
    """Do identical feature vectors carry conflicting labels? If so, quantify the ceiling.

    On a survey dataset many respondents give exactly the same answers to every question.
    When two such rows disagree about the outcome, no model that sees only these features
    can get both right — the features do not contain the information that separates them.
    That is irreducible (Bayes) error, and it caps accuracy no matter how good the model is.

    The bound reported here is the *best case*: for every group of identical rows, assume
    the model predicts that group's majority label. The minority rows inside conflicted
    groups are then necessarily wrong, so

        max_accuracy = 1 - (minority rows in conflicted groups) / n

    A model scoring below this is not necessarily leaving signal on the table, but a model
    appearing to score above it on this feature set would indicate something is wrong.
    """
    y = pd.Series(y).reset_index(drop=True)
    keys = pd.util.hash_pandas_object(X.reset_index(drop=True), index=False)
    grouped = y.groupby(keys.values)

    groups = pd.DataFrame({"n": grouped.size(), "pos": grouped.sum()})
    groups["neg"] = groups["n"] - groups["pos"]

    duplicated = groups[groups["n"] > 1]
    # A group is conflicted when it holds both labels.
    conflicted = duplicated[(duplicated["pos"] > 0) & (duplicated["neg"] > 0)]
    n_minority = int(conflicted[["pos", "neg"]].min(axis=1).sum())

    n = int(len(y))
    n_shared = int(duplicated["n"].sum())
    return {
        "n_rows": n,
        "n_unique_feature_vectors": int(len(groups)),
        "n_rows_sharing_a_feature_vector": n_shared,
        "pct_rows_sharing_a_feature_vector": round(100 * n_shared / n, 2),
        "n_duplicate_groups": int(len(duplicated)),
        "n_conflicted_groups": int(len(conflicted)),
        "n_rows_in_conflicted_groups": int(conflicted["n"].sum()),
        "n_unwinnable_rows": n_minority,
        "max_achievable_accuracy": round(1 - n_minority / n, 4),
    }


def duplicate_conflict_note(dup: dict, model_test_accuracy: float) -> str:
    """One honest sentence on how close the model is to the feature set's own ceiling."""
    if not dup or dup["n_conflicted_groups"] == 0:
        return (
            "No two rows share an identical feature vector with conflicting labels, so the "
            "features impose no measurable ceiling of this kind."
        )
    ceiling = dup["max_achievable_accuracy"]
    headroom = ceiling - model_test_accuracy
    base = (
        f"{dup['n_rows_sharing_a_feature_vector']:,} of {dup['n_rows']:,} rows "
        f"({dup['pct_rows_sharing_a_feature_vector']}%) share an identical feature vector "
        f"with at least one other row, and {dup['n_conflicted_groups']:,} of those groups "
        f"contain both outcome labels. {dup['n_unwinnable_rows']:,} rows are therefore "
        f"unwinnable for any model using only these features, capping accuracy at "
        f"{ceiling:.4f} against the measured {model_test_accuracy:.4f}."
    )
    # Don't let "irreducible error" become an excuse. It only explains the shortfall when
    # the model is actually near the ceiling.
    if headroom <= 0.02:
        return base + (
            " The model is close to that bound, so most of its remaining error is "
            "irreducible on this feature set rather than a modelling failure."
        )
    return base + (
        f" The bound is {headroom:.4f} above what the model achieves, so it does NOT "
        "explain the shortfall — duplicate-label conflict is a real but minor part of the "
        "error here, and the rest is the model, the features' weak association with the "
        "outcome, or both."
    )


def operating_point_note(metrics_050: dict, metrics_alt: dict) -> str:
    """Explain the threshold-0.5 row when 0.5 is a bad operating point.

    0.5 is a convention, not a decision rule. On a low-prevalence problem a *well
    calibrated* model should rarely output a probability above 0.5 — most people really
    do have a below-even chance — so thresholding there yields high specificity and very
    low recall. That looks like a broken model and is not one; it is the threshold that
    is wrong, not the ranking. Calibration makes this more visible, not worse: an
    uncalibrated class-weighted model inflates scores, which flatters recall at 0.5 while
    making the probability itself meaningless.

    Both rows are reported so the trade-off is visible. This note says which to read.
    """
    prevalence = metrics_050.get("prevalence")
    r050, ralt = metrics_050["recall_sensitivity"], metrics_alt["recall_sensitivity"]
    t_alt = metrics_alt["threshold"]
    base = (
        f"Positive-class prevalence is {prevalence:.4f}. At threshold 0.5 the model "
        f"recovers {r050:.4f} of positives at {metrics_050['specificity']:.4f} "
        f"specificity; at the sensitivity-oriented threshold {t_alt:.4f} it recovers "
        f"{ralt:.4f} at {metrics_alt['specificity']:.4f} specificity."
    )
    if ralt - r050 < 0.20:
        return base + " The two operating points are close, so 0.5 is a reasonable default here."
    return base + (
        f" The {ralt - r050:+.4f} swing in recall means **0.5 is not a meaningful "
        "operating point for this model**. A calibrated probability on a problem with "
        f"{prevalence:.1%} prevalence only exceeds 0.5 for the most extreme profiles, so "
        "thresholding there behaves as a high-precision screen and misses most positives. "
        "The low recall in the 0.5 row is a property of that arbitrary cut-off, not "
        "evidence that the model failed to rank patients — ROC-AUC and PR-AUC, which are "
        "threshold-free, are the metrics to compare against other models. Any real use "
        "would set the threshold from the cost of a false negative against a false alarm."
    )


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

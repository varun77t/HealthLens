"""Dataset profiling utilities (no plotting, no modelling).

Every function takes the loaded ``(X, y, spec)`` and returns a plain pandas object
or dict so results can be rendered in a notebook, serialised to JSON, or asserted
in tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data.feature_spec import FeatureSpec


def dataset_overview(X: pd.DataFrame, y: pd.Series, spec: FeatureSpec) -> dict:
    return {
        "disease": spec.disease,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_numeric": len(spec.numeric),
        "n_categorical": len(spec.categorical),
        "n_binary": len(spec.binary),
        "target": str(y.name),
        "target_positive_meaning": spec.notes.split(".")[0],
        "total_missing_cells": int(X.isna().sum().sum()),
        "rows_with_any_missing": int(X.isna().any(axis=1).sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "memory_mb": round(X.memory_usage(deep=True).sum() / 1e6, 3),
    }


def missing_value_table(X: pd.DataFrame) -> pd.DataFrame:
    n = len(X)
    rows = []
    for col in X.columns:
        miss = int(X[col].isna().sum())
        rows.append(
            {
                "column": col,
                "dtype": str(X[col].dtype),
                "n_missing": miss,
                "pct_missing": round(100 * miss / n, 2),
                "n_unique": int(X[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def target_distribution(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().sort_index()
    prop = y.value_counts(normalize=True).sort_index()
    df = pd.DataFrame({"class": counts.index, "count": counts.values, "proportion": prop.values.round(4)})
    return df.reset_index(drop=True)


def class_imbalance_ratio(y: pd.Series) -> float:
    counts = y.value_counts()
    return round(float(counts.max() / counts.min()), 2)


def numeric_summary(X: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    cols = spec.numeric
    if not cols:
        return pd.DataFrame()
    desc = X[cols].describe().T
    desc["missing"] = X[cols].isna().sum()
    desc["skew"] = X[cols].skew(numeric_only=True)
    return desc.round(3)


def categorical_summary(X: pd.DataFrame, spec: FeatureSpec) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for col in [*spec.categorical, *spec.binary]:
        vc = X[col].value_counts(dropna=False)
        vp = X[col].value_counts(dropna=False, normalize=True)
        out[col] = pd.DataFrame(
            {"value": vc.index.astype(str), "count": vc.values, "proportion": vp.values.round(4)}
        ).reset_index(drop=True)
    return out


def outlier_scan(X: pd.DataFrame, spec: FeatureSpec, k: float = 1.5) -> pd.DataFrame:
    """IQR-based outlier COUNT per numeric column.

    This flags values outside [Q1 - k*IQR, Q3 + k*IQR]. It is diagnostic only -
    medical measurements legitimately contain extreme values and nothing is removed
    on the basis of this scan.
    """
    rows = []
    for col in spec.numeric:
        s = X[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append(
            {
                "column": col,
                "q1": round(float(q1), 3),
                "q3": round(float(q3), 3),
                "iqr_lower_fence": round(float(lo), 3),
                "iqr_upper_fence": round(float(hi), 3),
                "n_outliers": n_out,
                "pct_outliers": round(100 * n_out / len(s), 2),
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3),
            }
        )
    return pd.DataFrame(rows)


def missingness_target_association(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Is a column's *missingness* itself predictive of the target?

    For every column that has missing values, compares the positive rate among rows
    where it is missing against rows where it is present.

    This matters because imputation happens inside the pipeline: if a lab is only
    ordered when a clinician already suspects the disease, then "this value is missing"
    carries diagnostic information, and a model can score well by exploiting the
    measurement pattern rather than the measurement. That is not train/test leakage —
    the split is still clean — but it means performance may not transfer to a setting
    where the test is ordered routinely. Reported so the effect can be judged, not
    silently corrected.
    """
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    overall = float(y.mean())
    rows = []
    for col in X.columns:
        mask = X[col].isna()
        n_missing = int(mask.sum())
        if n_missing == 0:
            continue
        present = y[~mask]
        rows.append(
            {
                "column": col,
                "n_missing": n_missing,
                "pct_missing": round(100 * n_missing / len(X), 2),
                "positive_rate_when_missing": round(float(y[mask].mean()), 4),
                "positive_rate_when_present": round(float(present.mean()), 4) if len(present) else float("nan"),
                "difference": round(float(y[mask].mean() - present.mean()), 4) if len(present) else float("nan"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["column", "n_missing", "pct_missing", "positive_rate_when_missing",
                     "positive_rate_when_present", "difference"]
        )
    out = pd.DataFrame(rows)
    out["overall_positive_rate"] = round(overall, 4)
    return out.sort_values("difference", key=abs, ascending=False).reset_index(drop=True)


def correlation_matrix(X: pd.DataFrame, spec: FeatureSpec, method: str = "spearman") -> pd.DataFrame:
    cols = spec.numeric
    if len(cols) < 2:
        return pd.DataFrame()
    return X[cols].corr(method=method).round(3)


def target_feature_association(X: pd.DataFrame, y: pd.Series, spec: FeatureSpec) -> pd.DataFrame:
    """Point-biserial-style correlation of each numeric feature with the binary target.

    Descriptive only (uses all rows, pairwise-complete). Not a model.
    """
    rows = []
    for col in spec.numeric:
        s = X[col]
        mask = s.notna()
        if mask.sum() < 3:
            continue
        corr = np.corrcoef(s[mask], y[mask])[0, 1]
        rows.append({"feature": col, "corr_with_target": round(float(corr), 3)})
    return pd.DataFrame(rows).sort_values("corr_with_target", key=abs, ascending=False).reset_index(drop=True)


def full_profile(X: pd.DataFrame, y: pd.Series, spec: FeatureSpec) -> dict:
    """Assemble a JSON-serialisable profile dict."""
    return {
        "overview": dataset_overview(X, y, spec),
        "class_imbalance_ratio": class_imbalance_ratio(y),
        "target_distribution": target_distribution(y).to_dict(orient="records"),
        "missing_values": missing_value_table(X).to_dict(orient="records"),
        "numeric_summary": numeric_summary(X, spec).reset_index().rename(columns={"index": "feature"}).to_dict(orient="records"),
        "outlier_scan": outlier_scan(X, spec).to_dict(orient="records"),
        "target_feature_association": target_feature_association(X, y, spec).to_dict(orient="records"),
        "notes": spec.notes,
    }

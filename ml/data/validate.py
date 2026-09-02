"""Structural validation of the loaded datasets.

These checks guard against silent loader breakage (renamed columns, wrong target
transform, an all-NaN coerced column). They assert *shape and semantics*, never
model performance.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Expected (rows, predictor-columns) after loading + target transform.
EXPECTED_SHAPE = {
    "heart": (303, 13),
    "kidney": (400, 24),
    "diabetes": (253_680, 21),
    "statlog": (270, 13),
}


@dataclass
class ValidationResult:
    disease: str
    n_rows: int
    n_features: int
    target_name: str
    target_counts: dict[int, int]
    positive_rate: float
    n_all_nan_columns: int
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def validate(disease: str, X: pd.DataFrame, y: pd.Series, spec) -> ValidationResult:
    problems: list[str] = []

    exp_rows, exp_feats = EXPECTED_SHAPE[disease]
    if X.shape[0] != exp_rows:
        problems.append(f"row count {X.shape[0]} != expected {exp_rows}")
    if X.shape[1] != exp_feats:
        problems.append(f"feature count {X.shape[1]} != expected {exp_feats}")

    if list(X.columns) != spec.features:
        problems.append("X columns do not match FeatureSpec.features order")

    if len(X) != len(y):
        problems.append(f"X/y length mismatch: {len(X)} vs {len(y)}")

    bad_target = set(pd.unique(y.dropna())) - {0, 1}
    if bad_target:
        problems.append(f"target has non-binary values: {sorted(bad_target)}")
    if y.isna().any():
        problems.append(f"target has {int(y.isna().sum())} missing values")

    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        problems.append(f"columns coerced entirely to NaN: {all_nan}")

    counts = y.value_counts().sort_index().to_dict()
    counts = {int(k): int(v) for k, v in counts.items()}
    pos_rate = float(y.mean())

    return ValidationResult(
        disease=disease,
        n_rows=int(X.shape[0]),
        n_features=int(X.shape[1]),
        target_name=str(y.name),
        target_counts=counts,
        positive_rate=pos_rate,
        n_all_nan_columns=len(all_nan),
        problems=problems,
    )

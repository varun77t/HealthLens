"""Phase 1 structural tests: loaders return the documented shapes, targets and roles.

These require the raw CSV cache (run `python -m ml.data.download_all` first). They
assert structure only — never model performance.
"""
from __future__ import annotations

import pytest

from ml.data.loaders import LOADERS, load
from ml.data.validate import EXPECTED_SHAPE, validate


@pytest.mark.parametrize("disease", list(LOADERS))
def test_loader_shape_and_target(disease):
    X, y, spec = load(disease)
    result = validate(disease, X, y, spec)
    assert result.ok, f"{disease}: {result.problems}"

    exp_rows, exp_feats = EXPECTED_SHAPE[disease]
    assert X.shape == (exp_rows, exp_feats)
    assert len(y) == exp_rows
    assert set(y.unique()) <= {0, 1}
    assert not y.isna().any()
    assert list(X.columns) == spec.features


@pytest.mark.parametrize("disease", list(LOADERS))
def test_no_column_fully_nan(disease):
    X, _, _ = load(disease)
    fully_nan = [c for c in X.columns if X[c].isna().all()]
    assert not fully_nan, f"{disease}: columns coerced entirely to NaN: {fully_nan}"


def test_heart_target_transform_matches_definition():
    """num > 0 -> 1; count must equal the raw file."""
    import pandas as pd

    from config import RAW_DATA_DIR

    raw = pd.read_csv(RAW_DATA_DIR / "heart" / "heart.csv")
    expected_pos = int((pd.to_numeric(raw["num"], errors="coerce") > 0).sum())
    _, y, _ = load("heart")
    assert int(y.sum()) == expected_pos


def test_diabetes_uses_full_imbalanced_release():
    _, y, _ = load("diabetes")
    assert len(y) == 253_680
    assert 0.10 < y.mean() < 0.20  # real prevalence, not the 50/50 balanced subset


def test_kidney_has_expected_missingness():
    X, _, _ = load("kidney")
    # The CKD dataset is known to be missing-heavy; guard against an accidental dropna.
    assert X.isna().sum().sum() > 500

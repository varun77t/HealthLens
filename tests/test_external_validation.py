"""Phase 6 tests: a candidate dataset must prove it is independent before it validates.

The unit tests use synthetic frames. The integration test at the end pins the real,
measured finding about Statlog — if a future change ever lets that dataset be reported as
external validation, it fails.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.external.dataset_overlap import (
    MAX_ACCEPTABLE_OVERLAP,
    dataset_overlap,
    row_keys,
    schema_compatibility,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=["a", "b", "c"])


# --- row keys: the dtype trap -------------------------------------------------------

def test_row_keys_match_across_int_and_float_dtypes():
    """The bug this guards against.

    The same record can arrive as int64 in one release and float64 in another. A naive
    string comparison sees "63" vs "63.0", finds zero matches, and reports a fully
    contaminated dataset as perfectly clean — a false negative in the most dangerous
    direction.
    """
    ints = pd.DataFrame({"a": [63, 67], "b": [145, 160]}).astype("int64")
    floats = pd.DataFrame({"a": [63.0, 67.0], "b": [145.0, 160.0]}).astype("float64")
    assert list(row_keys(ints)) == list(row_keys(floats))


def test_row_keys_distinguish_genuinely_different_rows():
    a = pd.DataFrame({"a": [1.0], "b": [2.0]})
    b = pd.DataFrame({"a": [1.0], "b": [2.001]})
    assert list(row_keys(a)) != list(row_keys(b))


def test_row_keys_handle_missing_values():
    a = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 2.0]})
    keys = row_keys(a)
    assert len(set(keys)) == 2
    assert "nan" in keys.iloc[1]


# --- overlap detection ----------------------------------------------------------------

def test_disjoint_datasets_are_independent():
    ref = _frame([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    cand = _frame([[7, 7, 7], [8, 8, 8]])
    r = dataset_overlap(cand, pd.Series([0, 1]), ref, pd.Series([0, 1, 0]))
    assert r.independent is True
    assert r.n_matched == 0
    assert r.pct_unseen == 100.0
    assert r.reasons == []
    assert "Independent" in r.verdict


def test_full_subset_is_rejected():
    ref = _frame([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    cand = _frame([[1, 1, 1], [2, 2, 2]])
    r = dataset_overlap(cand, pd.Series([0, 1]), ref, pd.Series([0, 1, 0, 1]))
    assert r.independent is False
    assert r.n_matched == 2 and r.pct_matched == 100.0
    assert r.n_unseen == 0
    assert "NOT INDEPENDENT" in r.verdict
    assert "subset of the training dataset" in r.verdict


def test_label_agreement_is_reported():
    """Matching features *and* labels means the same records, not similar patients."""
    ref = _frame([[1, 1, 1], [2, 2, 2]])
    cand = _frame([[1, 1, 1], [2, 2, 2]])
    same = dataset_overlap(cand, pd.Series([0, 1]), ref, pd.Series([0, 1]))
    assert same.n_label_agreements == 2
    assert any("same label" in r for r in same.reasons)

    flipped = dataset_overlap(cand, pd.Series([1, 0]), ref, pd.Series([0, 1]))
    assert flipped.n_label_agreements == 0
    assert not any("same label" in r for r in flipped.reasons)


def test_train_and_test_membership_are_separated():
    ref = _frame([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    train, test = ref.iloc[:3], ref.iloc[3:]
    cand = _frame([[1, 1, 1], [4, 4, 4]])
    r = dataset_overlap(cand, pd.Series([0, 1]), ref, pd.Series([0, 1, 0, 1]),
                        X_train=train, X_test=test)
    assert r.n_in_train == 1
    assert r.n_in_test == 1
    assert any("training split" in x for x in r.reasons)


def test_small_coincidental_overlap_stays_acceptable():
    ref = _frame([[i, i, i] for i in range(100)])
    cand = _frame([[0, 0, 0]] + [[i, i, i] for i in range(200, 299)])
    r = dataset_overlap(cand, pd.Series([0] * 100), ref, pd.Series([0] * 100))
    assert r.n_matched == 1
    assert r.pct_matched <= 100 * MAX_ACCEPTABLE_OVERLAP
    assert r.independent is True


def test_overlap_just_above_the_threshold_is_rejected():
    ref = _frame([[i, i, i] for i in range(100)])
    cand = _frame([[i, i, i] for i in range(10)] + [[i, i, i] for i in range(200, 290)])
    r = dataset_overlap(cand, pd.Series([0] * 100), ref, pd.Series([0] * 100))
    assert r.n_matched == 10 and r.pct_matched == 10.0
    assert r.independent is False


# --- schema compatibility ---------------------------------------------------------------

def test_schema_compatibility_accepts_matching_encodings():
    a = pd.DataFrame({"thal": [3.0, 6.0, 7.0], "age": [40.0, 50.0, 60.0]})
    b = pd.DataFrame({"thal": [3.0, 7.0, 7.0], "age": [45.0, 55.0, 58.0]})
    out = schema_compatibility(a, b, categorical=["thal"])
    assert out["compatible"].all()


def test_schema_compatibility_flags_a_recoded_categorical():
    """A 'thal' recoded 0/1/2 must not align silently against 3/6/7."""
    a = pd.DataFrame({"thal": [3.0, 6.0, 7.0]})
    b = pd.DataFrame({"thal": [0.0, 1.0, 2.0]})
    out = schema_compatibility(a, b, categorical=["thal"])
    assert not out["compatible"].all()
    assert "absent from the reference encoding" in out.iloc[0]["note"]


def test_schema_compatibility_flags_a_unit_change():
    a = pd.DataFrame({"chol": [126.0, 300.0, 564.0]})
    b = pd.DataFrame({"chol": [3.2, 7.8, 14.6]})  # mmol/L instead of mg/dl
    out = schema_compatibility(a, b, categorical=[])
    assert not out["compatible"].all()
    assert "different units" in out.iloc[0]["note"]


def test_schema_compatibility_flags_a_missing_column():
    a = pd.DataFrame({"thal": [3.0], "ca": [0.0]})
    b = pd.DataFrame({"thal": [3.0]})
    out = schema_compatibility(a, b, categorical=["thal", "ca"])
    assert not out["compatible"].all()
    assert not out.set_index("column").loc["ca", "in_both"]


# --- the real finding, pinned -------------------------------------------------------------

def test_statlog_is_not_independent_of_the_heart_training_data():
    """Regression test for the Phase 6 finding.

    Statlog (UCI 145) is a 270-row subset of the Cleveland database (UCI 45). If this
    ever starts passing as independent, something has silently changed and the heart
    model would gain a fabricated external validation.
    """
    from ml.data.loaders import load
    from ml.training.splits import make_split

    X_int, y_int, _ = load("heart")
    X_ext, y_ext, _ = load("statlog")
    split = make_split("heart", X_int, y_int)
    X_train, X_test, _, _ = split.apply(X_int, y_int)

    r = dataset_overlap(X_ext, y_ext, X_int, y_int, X_train=X_train, X_test=X_test,
                        columns=list(X_int.columns))
    assert r.independent is False
    assert r.n_matched == 270, "all 270 Statlog rows should match a Cleveland row"
    assert r.n_label_agreements == 270, "and carry the same label"
    assert r.n_unseen == 0
    assert r.n_in_train == 222


def test_statlog_encodings_are_compatible_with_cleveland():
    """The Phase 1 assumption, now verified rather than assumed.

    Compatibility is what makes the contamination finding meaningful: the rows really are
    the same records, not merely coincidentally equal under a mismatched encoding.
    """
    from ml.data.loaders import load

    X_int, _, spec = load("heart")
    X_ext, _, _ = load("statlog")
    out = schema_compatibility(
        X_int, X_ext, categorical=list(spec.categorical) + list(spec.binary)
    )
    assert out["compatible"].all(), out[~out["compatible"]].to_dict(orient="records")


def test_external_validation_run_reports_rejected():
    from ml.external.heart_statlog import run_external_validation

    result = run_external_validation(write=False)
    assert result["status"] == "rejected"
    assert result["usable_as_external_validation"] is False
    assert result["schema_compatible"] is True
    assert "NOT EXTERNAL VALIDATION" in result["metrics_interpretation"]

"""Tests for the Phase 5 machinery: candidate exclusion, the duplicate-label ceiling,
and the SMOTE-vs-class-weight comparison.

Unit tests on synthetic inputs — no disease model is trained here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.data.feature_spec import FeatureSpec
from ml.evaluation.diagnostics import (
    duplicate_conflict_note,
    duplicate_label_conflict_probe,
    operating_point_note,
)
from ml.training.imbalance import compare_imbalance_strategies, imbalance_note
from ml.training.model_zoo import (
    MODEL_NAMES,
    candidate_models,
    exclusion_reasons,
)
from ml.training.param_space import param_space


# --- candidate exclusion ------------------------------------------------------------

def test_diabetes_excludes_svc_with_a_reason():
    """SVC is intractable at 200k rows; it must be dropped *and* explained."""
    assert "svc" not in candidate_models("diabetes")
    reasons = exclusion_reasons("diabetes")
    assert "svc" in reasons
    assert len(reasons["svc"]) > 100, "an exclusion needs a real justification, not a label"


def test_small_diseases_keep_the_whole_zoo():
    for disease in ("heart", "kidney"):
        assert candidate_models(disease) == MODEL_NAMES
        assert exclusion_reasons(disease) == {}


def test_exclusion_reasons_are_a_copy():
    """Callers must not be able to mutate the registry."""
    exclusion_reasons("diabetes")["svc"] = "tampered"
    assert "tampered" not in exclusion_reasons("diabetes")["svc"]


def test_diabetes_narrows_the_random_forest_search_only():
    base = param_space("random_forest")
    narrowed = param_space("random_forest", disease="diabetes")
    assert base["clf__n_estimators"].args == (200, 800)
    assert narrowed["clf__n_estimators"].args == (150, 400)
    # Every other axis is untouched, and other models are untouched.
    assert set(base) == set(narrowed)
    assert param_space("xgboost", disease="diabetes") == param_space("xgboost")


def test_unknown_disease_leaves_the_space_alone():
    assert param_space("logreg", disease="nonexistent") == param_space("logreg")


# --- duplicate-label ceiling ---------------------------------------------------------

def test_no_duplicates_means_no_ceiling():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, 0, 1])
    d = duplicate_label_conflict_probe(X, y)
    assert d["n_duplicate_groups"] == 0
    assert d["n_conflicted_groups"] == 0
    assert d["max_achievable_accuracy"] == 1.0
    assert "no measurable ceiling" in duplicate_conflict_note(d, 0.9)


def test_duplicates_with_agreeing_labels_impose_no_ceiling():
    X = pd.DataFrame({"a": [1.0, 1.0, 2.0, 2.0]})
    y = pd.Series([1, 1, 0, 0])
    d = duplicate_label_conflict_probe(X, y)
    assert d["n_duplicate_groups"] == 2
    assert d["n_conflicted_groups"] == 0
    assert d["max_achievable_accuracy"] == 1.0


def test_conflicting_duplicates_cap_accuracy_at_the_majority_label():
    # One group of 4 identical rows: 3 positive, 1 negative. Best case gets 3 of 4.
    X = pd.DataFrame({"a": [1.0] * 4})
    y = pd.Series([1, 1, 1, 0])
    d = duplicate_label_conflict_probe(X, y)
    assert d["n_conflicted_groups"] == 1
    assert d["n_rows_in_conflicted_groups"] == 4
    assert d["n_unwinnable_rows"] == 1
    assert d["max_achievable_accuracy"] == 0.75


def test_evenly_split_duplicates_cost_half_the_group():
    X = pd.DataFrame({"a": [1.0] * 4 + [2.0] * 6})
    y = pd.Series([1, 1, 0, 0] + [0] * 6)
    d = duplicate_label_conflict_probe(X, y)
    assert d["n_unwinnable_rows"] == 2  # the minority half of the conflicted group
    assert d["max_achievable_accuracy"] == 0.8


def test_ceiling_uses_all_columns_not_just_one():
    """Rows differing in any column are not duplicates."""
    X = pd.DataFrame({"a": [1.0, 1.0], "b": [1.0, 2.0]})
    y = pd.Series([0, 1])
    assert duplicate_label_conflict_probe(X, y)["n_duplicate_groups"] == 0


def test_note_refuses_to_excuse_a_model_far_below_the_ceiling():
    """Irreducible error must not be used to explain away a large shortfall."""
    d = {"n_rows": 1000, "n_rows_sharing_a_feature_vector": 100,
         "pct_rows_sharing_a_feature_vector": 10.0, "n_conflicted_groups": 5,
         "n_unwinnable_rows": 10, "max_achievable_accuracy": 0.99}
    far = duplicate_conflict_note(d, 0.75)
    assert "does NOT" in far and "explain the shortfall" in far
    near = duplicate_conflict_note(d, 0.985)
    assert "close to that bound" in near


# --- operating point ------------------------------------------------------------------

def _row(recall, specificity, threshold, prevalence):
    return {"recall_sensitivity": recall, "specificity": specificity,
            "threshold": threshold, "prevalence": prevalence}


def test_operating_point_note_warns_when_0_5_is_a_bad_threshold():
    """The diabetes case: a calibrated model at 14% prevalence barely crosses 0.5."""
    note = operating_point_note(
        _row(0.1464, 0.9829, 0.5, 0.1383), _row(0.7984, 0.7090, 0.1388, 0.1383)
    )
    assert "not a meaningful operating point" in note
    assert "not evidence that the model failed to rank patients" in note


def test_operating_point_note_is_calm_when_0_5_is_fine():
    """The heart case: balanced prevalence, the two thresholds behave similarly."""
    note = operating_point_note(
        _row(0.8929, 0.8788, 0.5, 0.459), _row(0.8214, 0.9394, 0.6436, 0.459)
    )
    assert "reasonable default" in note
    assert "not a meaningful operating point" not in note


def test_operating_point_note_quotes_both_thresholds():
    note = operating_point_note(
        _row(0.20, 0.98, 0.5, 0.10), _row(0.85, 0.60, 0.1100, 0.10)
    )
    assert "0.1100" in note and "0.2000" in note and "0.8500" in note


# --- SMOTE vs class weights -----------------------------------------------------------

@pytest.fixture(scope="module")
def imbalanced_data():
    rng = np.random.default_rng(0)
    n = 400
    y = pd.Series((rng.random(n) < 0.2).astype(int))
    X = pd.DataFrame({
        "num1": rng.normal(size=n) + y * 0.9,
        "num2": rng.normal(size=n),
        "bin1": (rng.random(n) < 0.5).astype(int),
    })
    spec = FeatureSpec(
        disease="synthetic", target="t",
        numeric=["num1", "num2"], categorical=[], binary=["bin1"],
        descriptions={}, notes="",
    )
    return X, y, spec


def test_imbalance_comparison_reports_both_strategies(imbalanced_data):
    X, y, spec = imbalanced_data
    table = compare_imbalance_strategies("heart", "logreg", X, y, spec, n_splits=3)
    assert list(table["strategy"]) == ["class_weight", "smote"]
    for col in ("roc_auc_mean", "pr_auc_mean", "recall_mean", "pr_auc_std"):
        assert col in table.columns
        assert table[col].between(0, 1).all()


def test_imbalance_note_calls_a_small_difference_noise(imbalanced_data):
    X, y, spec = imbalanced_data
    table = compare_imbalance_strategies("heart", "logreg", X, y, spec, n_splits=3)
    note = imbalance_note(table, chosen="class_weight")
    assert "PR-AUC" in note
    assert "Shipped strategy: class_weight" in note


def test_imbalance_note_flags_a_real_degradation():
    table = pd.DataFrame([
        {"strategy": "class_weight", "pr_auc_mean": 0.80, "pr_auc_std": 0.005,
         "roc_auc_mean": 0.90, "recall_mean": 0.70},
        {"strategy": "smote", "pr_auc_mean": 0.60, "pr_auc_std": 0.005,
         "roc_auc_mean": 0.85, "recall_mean": 0.80},
    ])
    assert "a degradation" in imbalance_note(table)


def test_imbalance_note_flags_no_evidence_when_within_noise():
    table = pd.DataFrame([
        {"strategy": "class_weight", "pr_auc_mean": 0.800, "pr_auc_std": 0.05,
         "roc_auc_mean": 0.90, "recall_mean": 0.70},
        {"strategy": "smote", "pr_auc_mean": 0.805, "pr_auc_std": 0.05,
         "roc_auc_mean": 0.90, "recall_mean": 0.72},
    ])
    assert "no evidence it helps" in imbalance_note(table)

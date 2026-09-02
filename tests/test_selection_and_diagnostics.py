"""Tests for the honesty machinery: selection margin and performance-attribution probes.

These guard the parts of the pipeline whose job is to stop a result being over-read.
They are unit tests on synthetic inputs — no model is trained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.diagnostics import (
    attribution_note,
    complete_case_probe,
    missingness_only_probe,
)
from ml.evaluation.metrics import youden_threshold
from ml.training.experiment import MIN_MEANINGFUL_GAP, SATURATION_LEVEL, selection_margin


def _ranked(models, std):
    return pd.DataFrame({"model": models, "roc_auc_std": std})


# --- selection margin -------------------------------------------------------------

def test_noisy_gap_is_not_decisive():
    ranked = _ranked(["a", "b"], [0.03, 0.03])
    tuned = {"a": {"cv_roc_auc": 0.9079}, "b": {"cv_roc_auc": 0.9074}}
    m = selection_margin(ranked, tuned, "a")
    assert m["decisive"] is False
    assert any("standard deviation" in r for r in m["reasons_to_doubt"])


def test_saturated_metric_with_zero_std_is_not_decisive():
    """Regression test.

    An easy dataset makes every model score ~1.0 with a fold-to-fold std of exactly 0.0.
    A naive `gap > std` test reads that as a decisive win, when it is the least
    trustworthy case there is: the metric has simply run out of resolution.
    """
    ranked = _ranked(["svc", "random_forest"], [0.0, 0.0005])
    tuned = {"svc": {"cv_roc_auc": 1.0}, "random_forest": {"cv_roc_auc": 0.9998}}
    m = selection_margin(ranked, tuned, "svc")
    assert m["gap"] > m["cv_roc_auc_std"]  # the naive test would pass here
    assert m["decisive"] is False, "a saturated metric must never read as decisive"
    assert any("saturated" in r for r in m["reasons_to_doubt"])


def test_genuinely_decisive_gap_is_reported_as_such():
    ranked = _ranked(["a", "b"], [0.004, 0.004])
    tuned = {"a": {"cv_roc_auc": 0.90}, "b": {"cv_roc_auc": 0.82}}
    m = selection_margin(ranked, tuned, "a")
    assert m["decisive"] is True
    assert m["reasons_to_doubt"] == []


def test_small_gap_below_resolution_floor_is_not_decisive():
    ranked = _ranked(["a", "b"], [0.0, 0.0])
    tuned = {"a": {"cv_roc_auc": 0.800}, "b": {"cv_roc_auc": 0.7985}}
    m = selection_margin(ranked, tuned, "a")
    assert m["gap"] < MIN_MEANINGFUL_GAP
    assert m["decisive"] is False


def test_single_tuned_model_is_never_decisive():
    m = selection_margin(_ranked(["a"], [0.01]), {"a": {"cv_roc_auc": 0.9}}, "a")
    assert m["decisive"] is False
    assert m["runner_up"] is None


def test_saturation_level_is_below_one():
    assert 0.9 < SATURATION_LEVEL < 1.0


# --- threshold robustness ----------------------------------------------------------

def test_youden_threshold_never_returns_infinity():
    """roc_curve's first threshold is inf; returning it would zero out every prediction."""
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]  # perfectly separated
    t = youden_threshold(y_true, y_prob)
    assert np.isfinite(t)
    assert 0.0 <= t <= 1.0


def test_youden_threshold_on_constant_scores():
    t = youden_threshold([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert np.isfinite(t)
    assert 0.0 <= t <= 1.0


# --- missingness probes ------------------------------------------------------------

def _frame(missing_pattern, y):
    X = pd.DataFrame({"a": np.arange(len(y), dtype=float), "b": np.ones(len(y))})
    X.loc[missing_pattern, "a"] = np.nan
    return X, pd.Series(y)


def test_probe_returns_none_when_nothing_is_missing():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 1.0, 2.0, 2.0]})
    y = pd.Series([0, 1, 0, 1])
    assert missingness_only_probe(X, y, X, y) is None


def test_probe_detects_perfectly_informative_missingness():
    """When missingness *is* the label, the probe must report near-perfect AUC."""
    n = 60
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    X, ys = _frame(y == 1, y)  # column 'a' missing exactly for the positive class
    probe = missingness_only_probe(X, ys, X, ys)
    assert probe is not None
    assert probe["test_roc_auc"] == pytest.approx(1.0, abs=1e-6)
    assert probe["n_indicator_features"] == 1


def test_probe_finds_nothing_when_missingness_is_random():
    rng = np.random.default_rng(0)
    n = 200
    y = pd.Series(rng.integers(0, 2, size=n))
    X, ys = _frame(rng.random(n) < 0.3, y)
    probe = missingness_only_probe(X, ys, X, ys)
    assert probe["test_roc_auc"] < 0.70


def test_complete_case_probe_reports_the_shift():
    n = 100
    y = pd.Series([0] * 50 + [1] * 50)
    X, ys = _frame(y == 1, y)  # every positive row is incomplete
    cc = complete_case_probe(X, ys)
    assert cc["n_complete_cases"] == 50
    assert cc["pct_retained"] == 50.0
    assert cc["positive_rate_all"] == 0.5
    assert cc["positive_rate_complete_cases"] == 0.0


def test_attribution_note_escalates_with_probe_strength():
    strong = attribution_note({"test_roc_auc": 0.97, "test_pr_auc": 0.9,
                               "test_accuracy": 0.9, "n_indicator_features": 3}, 0.99)
    weak = attribution_note({"test_roc_auc": 0.55, "test_pr_auc": 0.5,
                             "test_accuracy": 0.5, "n_indicator_features": 3}, 0.99)
    assert "not as evidence of clinical usefulness" in strong
    assert "does not explain most of the performance" in weak
    assert "No values are missing" in attribution_note(None, 0.9)

"""Phase 2 tests: the metric functions are correct on hand-built arrays.

No model is involved — these feed known ``(y_true, y_prob)`` vectors and check the
arithmetic, so the evaluation layer is trustworthy before any real run uses it.
"""
from __future__ import annotations

import numpy as np
import pytest

from ml.evaluation.metrics import (
    classification_metrics,
    confusion_at,
    threshold_sweep,
    youden_threshold,
)


def test_perfect_separation_scores_one():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.2, 0.8, 0.99])
    m = classification_metrics(y_true, y_prob, threshold=0.5)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["recall_sensitivity"] == pytest.approx(1.0)
    assert m["specificity"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)


def test_confusion_counts_add_up():
    y_true = [0, 0, 1, 1, 1]
    y_prob = [0.1, 0.9, 0.9, 0.4, 0.8]  # one FP, one FN at t=0.5
    cm = confusion_at(y_true, y_prob, 0.5)
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert sum(cm.values()) == 5


def test_specificity_is_nan_when_no_negatives():
    m = classification_metrics([1, 1, 1], [0.6, 0.7, 0.8])
    assert np.isnan(m["specificity"])


def test_threshold_sweep_monotone_recall():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(size=200)
    sweep = threshold_sweep(y_true, y_prob)
    # Higher threshold -> fewer positives predicted -> recall cannot increase.
    assert (sweep["recall_sensitivity"].diff().dropna() <= 1e-9).all()


def test_youden_threshold_in_unit_interval():
    y_true = [0, 0, 1, 1, 0, 1]
    y_prob = [0.2, 0.4, 0.6, 0.9, 0.1, 0.55]
    t = youden_threshold(y_true, y_prob)
    assert 0.0 <= t <= 1.0


def test_brier_matches_manual():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.3, 0.9, 0.6, 0.2])
    expected = np.mean((y_prob - y_true) ** 2)
    assert classification_metrics(y_true, y_prob)["brier"] == pytest.approx(expected)

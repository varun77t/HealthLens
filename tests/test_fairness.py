"""Phase 7 tests: subgroup metrics, their uncertainty, and the two ways to over-read them.

The machinery exists to stop two opposite mistakes: calling noise a disparity on a 61-row
test set, and calling a trivial gap a disparity on a 50,961-row one. Both are tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.calibration import expected_calibration_error
from ml.fairness.subgroup_metrics import (
    MIN_CLASS_N,
    PRACTICAL_GAP,
    age_bands,
    auc_standard_error,
    available_attributes,
    disparity_assessment,
    subgroup_table,
    wilson_interval,
)


# --- Wilson intervals ------------------------------------------------------------------

def test_wilson_interval_matches_known_values():
    """Textbook check: 10 successes in 20 trials."""
    lo, hi = wilson_interval(10, 20)
    assert lo == pytest.approx(0.2993, abs=1e-3)
    assert hi == pytest.approx(0.7007, abs=1e-3)


def test_wilson_interval_stays_inside_the_unit_range_at_the_extremes():
    """The reason Wilson is used instead of the normal approximation.

    With 7 successes out of 7 the normal interval collapses to zero width at 1.0, which
    would report a subgroup of seven patients as perfectly measured.
    """
    lo, hi = wilson_interval(7, 7)
    assert 0.0 <= lo <= 1.0 and hi == pytest.approx(1.0)
    assert lo < 0.7, "an interval from 7 observations must stay wide"

    lo0, hi0 = wilson_interval(0, 5)
    assert lo0 == pytest.approx(0.0)
    assert 0.0 < hi0 < 1.0


def test_wilson_interval_narrows_as_n_grows():
    small = wilson_interval(50, 100)
    large = wilson_interval(5000, 10000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_handles_zero_n():
    lo, hi = wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_auc_standard_error_shrinks_with_more_cases():
    assert auc_standard_error(0.85, 500, 500) < auc_standard_error(0.85, 10, 10)
    assert np.isnan(auc_standard_error(0.85, 0, 10))


# --- age banding -------------------------------------------------------------------------

def test_diabetes_age_codes_are_banded_as_survey_levels_not_years():
    """BRFSS stores a 1-13 code. Treating it as years would put everyone under 45."""
    codes = pd.Series([1, 5, 8, 13])
    bands = age_bands(codes, "diabetes")
    assert list(bands) == ["18-34", "35-49", "50-64", "65+"]


def test_year_ages_are_banded_as_years():
    years = pd.Series([30, 50, 60, 70])
    assert list(age_bands(years, "heart")) == ["<45", "45-54", "55-64", "65+"]


def test_kidney_has_no_sex_attribute():
    """Its absence is reported, never worked around."""
    X = pd.DataFrame({"age": [40.0, 60.0], "hemo": [12.0, 9.0]})
    attrs = available_attributes(X, "kidney")
    assert "sex" not in attrs
    assert "age_band" in attrs


# --- subgroup table ------------------------------------------------------------------------

def _synthetic(n_per_group, recall_a=0.9, recall_b=0.9, seed=0):
    """Two groups with controllable recall, for testing the verdicts."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, rec in [("a", recall_a), ("b", recall_b)]:
        for i in range(n_per_group):
            pos = i % 2 == 0
            if pos:
                p = 0.9 if rng.random() < rec else 0.1
            else:
                p = 0.1 if rng.random() < 0.9 else 0.9
            rows.append({"group": name, "y": int(pos), "p": p})
    df = pd.DataFrame(rows)
    return df["y"], df["p"], df["group"]


def test_small_subgroup_is_flagged_unreliable():
    y, p, g = _synthetic(8)
    table = subgroup_table(y, p, g)
    assert not table["reliable"].any()
    assert all("min" in c for c in table["caveat"])


def test_large_subgroup_is_marked_reliable():
    y, p, g = _synthetic(200)
    table = subgroup_table(y, p, g)
    assert table["reliable"].all()
    assert (table["caveat"] == "").all()


def test_table_reports_counts_and_intervals():
    y, p, g = _synthetic(100)
    table = subgroup_table(y, p, g)
    for _, row in table.iterrows():
        assert row["n"] == row["n_positive"] + row["n_negative"]
        assert row["recall_lo"] <= row["recall"] <= row["recall_hi"]
        assert row["roc_auc_lo"] <= row["roc_auc"] <= row["roc_auc_hi"]


def test_subgroup_with_one_class_has_no_auc():
    y = pd.Series([1, 1, 1, 1])
    p = pd.Series([0.9, 0.8, 0.7, 0.6])
    g = pd.Series(["a"] * 4)
    table = subgroup_table(y, p, g)
    assert np.isnan(table.iloc[0]["roc_auc"])
    assert not table.iloc[0]["reliable"]


def test_threshold_changes_rate_metrics_but_not_auc():
    """The core reason both thresholds are reported."""
    y, p, g = _synthetic(200)
    at_half = subgroup_table(y, p, g, threshold=0.5)
    at_low = subgroup_table(y, p, g, threshold=0.05)
    assert list(at_half["roc_auc"]) == list(at_low["roc_auc"])
    assert list(at_half["recall"]) != list(at_low["recall"])


# --- disparity verdicts ----------------------------------------------------------------------

def _table(vals, los, his, reliable=True, n=1000):
    return pd.DataFrame({
        "subgroup": ["a", "b"],
        "recall": vals, "recall_lo": los, "recall_hi": his,
        "reliable": [reliable, reliable], "n": [n, n],
    })


def test_overlapping_intervals_are_inconclusive():
    t = _table([0.90, 0.70], [0.75, 0.55], [0.97, 0.85])
    d = disparity_assessment(t, "recall")
    assert d["conclusive"] is False
    assert d["intervals_disjoint"] is False
    assert "consistent with no difference" in d["verdict"]


def test_unreliable_subgroups_are_never_conclusive_however_large_the_gap():
    t = _table([0.99, 0.10], [0.95, 0.02], [1.00, 0.30], reliable=False)
    d = disparity_assessment(t, "recall")
    assert d["intervals_disjoint"] is True
    assert d["conclusive"] is False
    assert "NOT evidence of a disparity" in d["verdict"]


def test_disjoint_but_tiny_gap_is_statistically_clear_yet_not_material():
    """The large-sample failure mode: narrow intervals make trivia significant."""
    t = _table([0.8200, 0.7900], [0.8150, 0.7850], [0.8250, 0.7950], n=50000)
    d = disparity_assessment(t, "recall")
    assert d["intervals_disjoint"] is True
    assert d["statistically_clear"] is True
    assert d["gap"] < PRACTICAL_GAP
    assert d["conclusive"] is False
    assert "not as a disparity worth acting on" in d["verdict"]


def test_disjoint_and_large_gap_is_material():
    t = _table([0.90, 0.60], [0.87, 0.56], [0.93, 0.64])
    d = disparity_assessment(t, "recall")
    assert d["conclusive"] is True
    assert "real and material difference" in d["verdict"]


def test_threshold_dependent_metrics_carry_the_threshold_caveat():
    t = _table([0.90, 0.60], [0.87, 0.56], [0.93, 0.64])
    assert "fixed decision threshold" in disparity_assessment(t, "recall")["verdict"]


def test_single_subgroup_cannot_be_compared():
    t = pd.DataFrame({"subgroup": ["a"], "recall": [0.9], "recall_lo": [0.8],
                      "recall_hi": [0.95], "reliable": [True], "n": [100]})
    d = disparity_assessment(t, "recall")
    assert d["conclusive"] is False
    assert "Fewer than two subgroups" in d["verdict"]


# --- expected calibration error ----------------------------------------------------------------

def test_ece_is_zero_for_perfectly_calibrated_predictions():
    """Half the rows at p=0.0 all negative, half at p=1.0 all positive."""
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.0] * 50 + [1.0] * 50)
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_systematic_overconfidence():
    y = np.array([0] * 90 + [1] * 10)   # true rate 0.10
    p = np.full(100, 0.9)               # claims 0.90
    assert expected_calibration_error(y, p) == pytest.approx(0.8, abs=1e-6)


def test_ece_is_blind_to_ranking():
    """Why ECE is reported next to Brier and never instead of it.

    Predicting the base rate for everyone is useless but almost perfectly calibrated.
    """
    y = np.array([0] * 80 + [1] * 20)
    base_rate = np.full(100, 0.2)
    assert expected_calibration_error(y, base_rate) == pytest.approx(0.0, abs=1e-9)


def test_ece_handles_empty_input():
    assert np.isnan(expected_calibration_error([], []))


# --- the real findings, pinned ---------------------------------------------------------------

def test_heart_subgroups_are_too_small_to_support_claims():
    """Regression test for the Phase 7 finding on heart (61 test patients)."""
    import json
    from pathlib import Path

    path = Path("reports/heart/fairness/disparities.json")
    if not path.exists():
        pytest.skip("fairness report not generated yet")
    r = json.loads(path.read_text(encoding="utf-8"))
    assert r["n_reliable_subgroups"] < r["n_subgroups"]
    for attr, per_threshold in r["disparities"].items():
        for tname, per_metric in per_threshold.items():
            for metric, d in per_metric.items():
                assert not d.get("conclusive"), (
                    f"heart {attr}/{metric}@{tname} was reported conclusive on a "
                    f"{r['test_n']}-row test set"
                )


def test_kidney_reports_sex_as_unavailable():
    import json
    from pathlib import Path

    path = Path("reports/kidney/fairness/disparities.json")
    if not path.exists():
        pytest.skip("fairness report not generated yet")
    r = json.loads(path.read_text(encoding="utf-8"))
    assert "sex" not in r["attributes_analysed"]
    assert any(a["attribute"] == "sex" for a in r["attributes_unavailable"])

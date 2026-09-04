"""v1 intake tests: the metadata the guided form is generated from.

The form is not hand-written — every input, label, option list and bound comes from
``GET /models/{disease}/schema``, which is built from ``serving.json``. That makes these
assets load-bearing UI: a field with no group never renders, an ordinal code with no options
becomes a box asking someone to type "8" for their age, and a tier that drifts from the SHAP
output mislabels which fields matter. Each of those is tested here.
"""
from __future__ import annotations

import json

import pytest

from backend.main import app
from backend.schemas.features import serving_spec
from config import MODELS_DIR, REPORTS_DIR
from ml.serving.field_groups import FIELD_GROUPS, FIELD_LABELS, GROUPS, options_for

DISEASES = ("heart", "kidney", "diabetes")


# --- presentation metadata covers the model exactly --------------------------------------


@pytest.mark.parametrize("disease", DISEASES)
def test_every_feature_has_a_group_and_a_label(disease):
    """An ungrouped field would silently never appear on the form."""
    names = {f["name"] for f in serving_spec(disease)["features"]}
    assert set(FIELD_GROUPS[disease]) == names
    assert set(FIELD_LABELS[disease]) == names


@pytest.mark.parametrize("disease", DISEASES)
def test_every_group_used_is_defined(disease):
    assert set(FIELD_GROUPS[disease].values()) <= set(GROUPS)


@pytest.mark.parametrize("disease", DISEASES)
def test_groups_are_ordered_and_non_empty(disease):
    groups = serving_spec(disease)["groups"]
    assert groups, f"{disease} exposes no groups"
    assert [g["order"] for g in groups] == sorted(g["order"] for g in groups)
    used = {f["group"] for f in serving_spec(disease)["features"]}
    assert {g["id"] for g in groups} == used


# --- coded fields are chosen, never typed --------------------------------------------------


@pytest.mark.parametrize("disease", DISEASES)
def test_categorical_and_binary_fields_all_offer_options(disease):
    for f in serving_spec(disease)["features"]:
        if f["kind"] in ("categorical", "binary"):
            assert f["options"], f"{disease}.{f['name']} has no options to pick from"
            assert {o["value"] for o in f["options"]} == set(f["categories"])


def test_ordinal_survey_codes_render_as_labelled_choices():
    """Age/GenHlth/Education/Income are numeric to the model but must be picked, not typed.

    Regression on a real defect: because these are modelled as numeric they initially had
    no options, so the form asked people to enter `8` for their age band.
    """
    feats = {f["name"]: f for f in serving_spec("diabetes")["features"]}
    for name, first, count in [
        ("Age", "18-24", 13),
        ("GenHlth", "Excellent", 5),
        ("Education", "Never attended school or only kindergarten", 6),
        ("Income", "Less than $10,000", 8),
    ]:
        opts = feats[name]["options"]
        assert opts and len(opts) == count, f"{name} should offer {count} labelled levels"
        assert opts[0]["label"] == first
        assert all(o["label"] != f"{o['value']:g}" for o in opts), f"{name} shows raw codes"


def test_continuous_fields_are_not_turned_into_dropdowns():
    feats = {f["name"]: f for f in serving_spec("diabetes")["features"]}
    for name in ("BMI", "MentHlth", "PhysHlth"):
        assert feats[name]["options"] is None


def test_an_unlabelled_ordinal_level_is_refused_not_dropped():
    """A missing label would make a real training value unselectable, so it raises."""
    with pytest.raises(KeyError, match="no label"):
        options_for("diabetes", "GenHlth", None, observed=(1.0, 9.0))


# --- tiers and coverage weights ------------------------------------------------------------


@pytest.mark.parametrize("disease", DISEASES)
def test_tiers_follow_the_models_own_shap_ranking(disease):
    """Core is a prefix of the SHAP ranking, not a hand-picked set."""
    feats = sorted(serving_spec(disease)["features"], key=lambda f: f["shap_rank"])
    tiers = [f["tier"] for f in feats]
    assert "core" in tiers
    # Once optional starts it never reverts to core.
    assert tiers == sorted(tiers, key=lambda t: 0 if t == "core" else 1)


@pytest.mark.parametrize("disease", DISEASES)
def test_core_tier_covers_most_of_the_attribution_mass(disease):
    feats = serving_spec(disease)["features"]
    core = sum(f["shap_mass_share"] for f in feats if f["tier"] == "core")
    assert 0.80 <= core < 1.0, f"{disease} core mass {core:.3f} outside the intended band"
    assert len([f for f in feats if f["tier"] == "core"]) < len(feats)


@pytest.mark.parametrize("disease", DISEASES)
def test_mass_shares_sum_to_one_and_match_the_shap_report(disease):
    feats = serving_spec(disease)["features"]
    assert sum(f["shap_mass_share"] for f in feats) == pytest.approx(1.0)
    report = json.loads(
        (REPORTS_DIR / disease / "shap_global.json").read_text(encoding="utf-8")
    )["global_importance"]
    ranks = {r["feature"]: i for i, r in enumerate(report, start=1)}
    for f in feats:
        assert f["shap_rank"] == ranks[f["name"]]


# --- schema endpoint -------------------------------------------------------------------------


@pytest.mark.parametrize("disease", DISEASES)
def test_schema_endpoint_carries_everything_the_form_needs(client, disease):
    body = client.get(f"/models/{disease}/schema").json()
    assert body["groups"] and body["tier_note"] and body["range_note"]
    for f in body["features"]:
        assert f["label"] and f["group"] and f["tier"] in ("core", "optional")


def test_diabetes_label_source_admits_which_income_levels_are_unverified(client):
    """The UCI metadata pins only three Income anchors; the rest must not be claimed."""
    note = client.get("/models/diabetes/schema").json()["value_label_source"]
    assert "not independently verified" in note
    assert "INCOME2" in note


# --- sample cases -----------------------------------------------------------------------------


@pytest.mark.parametrize("disease", DISEASES)
def test_samples_are_real_held_out_rows_with_both_outcomes(client, disease):
    body = client.get(f"/samples/{disease}").json()
    labels = {c["recorded_label"] for c in body["cases"]}
    assert labels == {0, 1}, "examples must show both outcomes, not only positives"
    assert "held-out test split" in body["note"]
    assert "not patients of this system" in body["note"]
    order = serving_spec(disease)["feature_order"]
    for c in body["cases"]:
        assert set(c["features"]) == set(order)


@pytest.mark.parametrize("disease", DISEASES)
def test_a_sample_case_can_be_predicted_as_submitted(client, disease):
    """The example must flow through /predict unchanged — no client-side fixing up."""
    case = client.get(f"/samples/{disease}").json()["cases"][0]
    r = client.post(f"/predict/{disease}", json=case["features"])
    assert r.status_code == 200, r.text
    assert 0.0 <= r.json()["probability"] <= 1.0


def test_samples_404_for_an_unknown_disease(client):
    assert client.get("/samples/liver").status_code == 404


def test_sample_rows_are_not_training_rows(client):
    """A sample drawn from the training split would be a rehearsal, not a demonstration."""
    from ml.data.loaders import load
    from ml.training.splits import make_split

    X, y, _ = load("heart")
    split = make_split("heart", X, y)
    _, X_test, _, _ = split.apply(X, y)
    test_rows = {
        tuple(None if v != v else round(float(v), 6) for v in row)
        for row in X_test[list(X_test.columns)].to_numpy()
    }
    cases = json.loads(
        (MODELS_DIR / "heart" / "samples.json").read_text(encoding="utf-8")
    )["cases"]
    order = serving_spec("heart")["feature_order"]
    for c in cases:
        key = tuple(
            None if c["features"][n] is None else round(float(c["features"][n]), 6)
            for n in order
        )
        assert key in test_rows, f"{c['id']} is not a held-out row"

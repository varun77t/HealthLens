"""Phase 8 tests: the API serves the trained models faithfully and carries the caveats.

Two categories here. The first is ordinary contract testing — status codes, shapes, ranges.
The second, and the reason most of this file exists, pins the findings from Phases 3-7 to
the endpoints, so a future change cannot quietly undo them:

* the risk band must agree with the model's own decision (the old fixed 0.33/0.66 grid
  labelled every diabetes case the model flags as "low");
* 0.5 must never be used as the diabetes decision point;
* a response must state that the model has no external validation;
* an unknown categorical level must be rejected, not one-hot encoded as all-zeros;
* the API must serve the metrics that were actually written to disk, not recomputed ones.

The registry is loaded once per module: three pipelines plus a KernelExplainer cost ~7 s.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.features import example_payload, serving_spec
from config import MODELS_DIR, REPORTS_DIR

DISEASES = ("heart", "kidney", "diabetes")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- service ---------------------------------------------------------------------------


def test_health_reports_every_module_loaded_and_explaining(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    for disease in DISEASES:
        assert body["diseases"][disease]["loaded"] is True
        assert body["diseases"][disease]["explainer_available"] is True


def test_root_and_health_carry_the_disclaimer(client):
    for path in ("/", "/health"):
        assert "not be interpreted as medical diagnosis" in client.get(path).json()["disclaimer"]


def test_models_lists_three_independent_modules(client):
    cards = client.get("/models").json()
    assert sorted(c["disease"] for c in cards) == sorted(DISEASES)
    # Three separate models, never a combined one.
    assert len({c["module"] for c in cards}) == 3
    for card in cards:
        assert card["limitations"], f"{card['disease']} card has no limitations listed"


# --- predictions -----------------------------------------------------------------------


@pytest.mark.parametrize("disease", DISEASES)
def test_predict_returns_a_probability_and_an_explanation(client, disease):
    r = client.post(f"/predict/{disease}", json=example_payload(disease))
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["explanation"]["contributions"], "no SHAP contributions returned"
    assert len(body["explanation"]["contributions"]) == body["n_features_expected"]
    assert "not be interpreted as medical diagnosis" in body["disclaimer"]


@pytest.mark.parametrize("disease", DISEASES)
def test_flag_follows_the_models_own_threshold_not_one_half(client, disease):
    body = client.post(f"/predict/{disease}", json=example_payload(disease)).json()
    assert body["flagged"] == (body["probability"] >= body["threshold"])
    assert "Youden" in body["threshold_rule"]


def test_diabetes_decision_is_not_taken_at_one_half(client):
    """Finding from Phase 5: at 0.5 this model's test recall is 0.1464, not 0.7984."""
    body = client.post("/predict/diabetes", json=example_payload("diabetes")).json()
    assert body["threshold"] == pytest.approx(0.13882786532243094)
    assert body["threshold"] < 0.5


@pytest.mark.parametrize("disease", DISEASES)
def test_risk_band_cannot_contradict_the_models_decision(client, disease):
    """The defect the old fixed bands had: 'low risk' on a case the model flags.

    With bands anchored on the operating threshold, `flagged` and the 'high' band are the
    same boundary by construction — so this is a regression test on that construction.
    """
    body = client.post(f"/predict/{disease}", json=example_payload(disease)).json()
    assert body["flagged"] == (body["risk_band"]["label"] == "high")
    assert body["risk_band"]["note"].startswith("Presentation buckets, not clinical")


def test_diabetes_bands_are_not_the_old_fixed_grid():
    """Regression on the measured defect, checked against the exported artifact.

    Under the fixed 0.33/0.66 grid, 44,243 of 50,961 diabetes test rows (86.8%) fell in
    "low" — including every case the model flags at 0.1388.
    """
    bands = serving_spec("diabetes")["risk_bands"]["bands"]
    assert [b["label"] for b in bands] == ["low", "moderate", "high"]
    assert bands[2]["min"] == pytest.approx(0.13882786532243094)
    assert bands[2]["min"] < 0.33, "the 'high' band must start at the operating threshold"


@pytest.mark.parametrize("disease", DISEASES)
def test_every_prediction_states_the_external_validation_position(client, disease):
    """No module has external validation; heart's attempt was rejected. Both must be said."""
    body = client.post(f"/predict/{disease}", json=example_payload(disease)).json()
    assert any("external validation" in w.lower() for w in body["warnings"])


def test_heart_reports_its_external_validation_as_rejected(client):
    card = next(c for c in client.get("/models").json() if c["disease"] == "heart")
    assert card["external_validation"]["status"] == "rejected"
    body = client.post("/predict/heart", json=example_payload("heart")).json()
    assert any("NO external validation" in w for w in body["warnings"])


def test_explanation_is_labelled_as_explaining_the_uncalibrated_model(client):
    """SHAP runs on the base pipeline; the probability comes from the calibrated one."""
    body = client.post("/predict/heart", json=example_payload("heart")).json()
    exp = body["explanation"]
    assert exp["explainer"] == "tree"
    assert "uncalibrated" in exp["note"].lower()
    assert 0.0 <= exp["uncalibrated_probability"] <= 1.0
    assert "uncalibrated_probability" in exp


def test_explanation_can_be_switched_off(client):
    """Kidney explanations cost ~0.8 s (KernelExplainer over an SVC); callers may skip."""
    r = client.post(
        "/predict/kidney",
        params={"include_explanation": "false"},
        json=example_payload("kidney"),
    )
    assert r.status_code == 200
    assert r.json()["explanation"] is None
    assert r.json()["probability"] is not None


def test_kidney_explainer_cost_is_published_not_hidden(client):
    card = next(c for c in client.get("/models").json() if c["disease"] == "kidney")
    assert card["explainer"]["kind"] == "kernel"
    assert card["explainer"]["explain_one_median_ms"] > 100


def test_diabetes_prediction_carries_its_age_band_error_profile(client):
    """Phase 7: one global threshold behaves very differently by age band."""
    body = client.post("/predict/diabetes", json=example_payload("diabetes")).json()
    assert any("age band" in w for w in body["warnings"])


def test_heart_does_not_claim_subgroup_performance_it_could_not_measure(client):
    """Only 1 of 6 heart subgroups was reliable on 61 test rows; none may be asserted."""
    body = client.post("/predict/heart", json=example_payload("heart")).json()
    assert not any("age band at this same threshold" in w for w in body["warnings"])


# --- input validation --------------------------------------------------------------------


def test_unknown_category_is_rejected_not_silently_zero_encoded(client):
    """`handle_unknown='ignore'` would encode cp=9 as all-zeros and score it confidently."""
    r = client.post("/predict/heart", json={**example_payload("heart"), "cp": 9.0})
    assert r.status_code == 422
    assert "cp" in json.dumps(r.json())


def test_value_far_outside_the_training_envelope_is_rejected(client):
    r = client.post("/predict/heart", json={**example_payload("heart"), "age": 500.0})
    assert r.status_code == 422


def test_value_outside_the_training_range_is_accepted_but_flagged(client):
    """Age 90 is plausible and unseen: the model has no support, which is a caveat not an error."""
    r = client.post("/predict/heart", json={**example_payload("heart"), "age": 90.0})
    assert r.status_code == 200
    flagged = r.json()["extrapolated_features"]
    assert [f["feature"] for f in flagged] == ["age"]
    assert any("training range" in w for w in r.json()["warnings"])


def test_unknown_field_is_rejected(client):
    assert client.post("/predict/heart", json={"cholesterol": 200}).status_code == 422


def test_empty_request_is_refused_rather_than_scored_from_medians(client):
    r = client.post("/predict/heart", json={})
    assert r.status_code == 422
    assert "no case" in json.dumps(r.json()).lower()


def test_partial_input_is_answered_and_the_imputation_reported(client):
    r = client.post("/predict/heart", json={"age": 60.0, "sex": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert body["n_features_provided"] == 2
    assert len(body["imputed_features"]) == body["n_features_expected"] - 2
    assert any("imputed" in w for w in body["warnings"])


def test_wrong_diseases_features_are_rejected(client):
    """The modules are independent; kidney features must not score on the heart model."""
    assert client.post("/predict/heart", json=example_payload("kidney")).status_code == 422


# --- scenario ------------------------------------------------------------------------------


def test_scenario_returns_both_estimates_and_the_illustrative_label(client):
    r = client.post(
        "/scenario/diabetes",
        json={"features": example_payload("diabetes"), "overrides": {"BMI": 40.0}},
    )
    assert r.status_code == 200
    body = r.json()
    assert "Illustrative model scenario" in body["label"]
    assert body["baseline"]["probability"] != body["modified"]["probability"]
    assert body["probability_delta"] == pytest.approx(
        body["modified"]["probability"] - body["baseline"]["probability"]
    )
    assert body["changed_features"][0]["feature"] == "BMI"


def test_scenario_warns_when_a_non_actionable_feature_is_changed(client):
    """GenHlth is plausibly a consequence of diabetes, so a 'what if' on it misleads."""
    body = client.post(
        "/scenario/diabetes",
        json={"features": example_payload("diabetes"), "overrides": {"GenHlth": 1.0}},
    ).json()
    assert any("GenHlth" in w and "consequence" in w for w in body["warnings"])


def test_scenario_rejects_an_out_of_range_override(client):
    r = client.post(
        "/scenario/heart",
        json={"features": example_payload("heart"), "overrides": {"thal": 99.0}},
    )
    assert r.status_code == 422
    assert "overrides" in json.dumps(r.json())


def test_scenario_requires_an_override(client):
    r = client.post(
        "/scenario/heart", json={"features": example_payload("heart"), "overrides": {}}
    )
    assert r.status_code == 422


# --- analytics ------------------------------------------------------------------------------


@pytest.mark.parametrize("disease", DISEASES)
def test_analytics_serves_the_metrics_that_were_written_to_disk(client, disease):
    """No metric is recomputed at request time; the API cannot invent a number."""
    served = client.get(f"/analytics/{disease}").json()
    on_disk = json.loads(
        (REPORTS_DIR / disease / "metrics.json").read_text(encoding="utf-8")
    )
    assert served["metrics"] == on_disk
    assert served["model_comparison"], "model comparison missing"
    assert served["shap_global_importance"], "global SHAP importance missing"


def test_analytics_serves_a_figure_and_refuses_a_path_outside_the_directory(client):
    figures = client.get("/analytics/heart").json()["figures"]
    assert "roc_curve.png" in figures
    assert client.get("/analytics/heart/figures/roc_curve.png").status_code == 200
    assert client.get("/analytics/heart/figures/..%2Fmetrics.json").status_code == 404


def test_unknown_disease_is_a_404_that_names_the_alternatives(client):
    r = client.get("/analytics/liver")
    assert r.status_code == 404
    assert "heart" in json.dumps(r.json())


# --- schema endpoint ----------------------------------------------------------------------


@pytest.mark.parametrize("disease", DISEASES)
def test_schema_endpoint_matches_the_model_pipeline(client, disease):
    body = client.get(f"/models/{disease}/schema").json()
    names = [f["name"] for f in body["features"]]
    assert names == body["feature_order"]
    assert len(names) == len(set(names))
    assert all(f["description"] for f in body["features"])


@pytest.mark.parametrize("disease", DISEASES)
def test_serving_ranges_are_declared_as_training_derived_not_clinical(disease):
    note = serving_spec(disease)["range_note"]
    assert "TRAINING" in note
    assert "not clinical" in note


@pytest.mark.parametrize("disease", DISEASES)
def test_serving_assets_are_in_step_with_the_model_card(disease):
    """The exported threshold and the card's risk bands must be the same number."""
    serving = serving_spec(disease)
    card = json.loads(
        (MODELS_DIR / disease / "metadata.json").read_text(encoding="utf-8")
    )
    assert card["risk_bands"]["operating_threshold"] == pytest.approx(
        serving["operating_threshold"]
    )
    assert card["model"]["algorithm"] == serving["model"]
    assert card["model"]["calibration"] == serving["calibration"]

"""Phase 3+ tests: shipped model artifacts are real, loadable and self-consistent.

Parametrised over every disease that has a trained model on disk, so Phases 4 and 5
are covered automatically once their pipelines exist. Diseases without artifacts are
skipped rather than failed — an untrained disease is not a broken one.

The central test here is :func:`test_reported_metrics_match_a_fresh_evaluation`: it
reloads the serialised pipeline, re-scores the held-out test split and asserts the
numbers equal what ``metrics.json`` claims. A hand-written or stale metric cannot
survive it.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from config import MODELS_DIR, REPORTS_DIR, risk_band
from ml.data.loaders import load
from ml.evaluation.metrics import classification_metrics
from ml.explainability.shap_explainer import DiseaseExplainer
from ml.training.splits import make_split

DISEASES = ["heart", "kidney", "diabetes"]


def _artifacts(disease: str):
    pipeline_path = MODELS_DIR / disease / "pipeline.joblib"
    meta_path = MODELS_DIR / disease / "metadata.json"
    metrics_path = REPORTS_DIR / disease / "metrics.json"
    if not (pipeline_path.exists() and meta_path.exists() and metrics_path.exists()):
        pytest.skip(f"{disease} has no trained model yet")
    return pipeline_path, meta_path, metrics_path


@pytest.fixture
def trained(request, load_dataset):
    disease = request.param if hasattr(request, "param") else None
    pipeline_path, meta_path, metrics_path = _artifacts(disease)
    X, y, spec = load_dataset(disease)
    split = make_split(disease, X, y)
    X_train, X_test, y_train, y_test = split.apply(X, y)
    return {
        "disease": disease,
        "spec": spec,
        "pipeline": joblib.load(pipeline_path),
        "base_pipeline": joblib.load(MODELS_DIR / disease / "base_pipeline.joblib"),
        "metadata": json.loads(meta_path.read_text(encoding="utf-8")),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def _param(fn):
    return pytest.mark.parametrize("trained", DISEASES, indirect=True)(fn)


# --- the pipeline actually works -------------------------------------------------

@_param
def test_predict_proba_is_a_valid_probability(trained):
    proba = trained["pipeline"].predict_proba(trained["X_test"])
    assert proba.shape == (len(trained["X_test"]), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


@_param
def test_risk_band_covers_every_prediction(trained):
    threshold = trained["metrics"]["test_set_alternative_threshold"]["threshold"]
    p = trained["pipeline"].predict_proba(trained["X_test"])[:, 1]
    assert all(risk_band(float(v), threshold) in {"low", "moderate", "high"} for v in p)


@_param
def test_risk_band_agrees_with_the_models_own_decision(trained):
    """A case the model flags must never be labelled anything but 'high'.

    Under the fixed 0.33/0.66 bands this failed badly for diabetes: the operating
    threshold is 0.1388, so all 50,961 test rows — including every true positive the model
    caught — were labelled 'low' or 'moderate'. Anchoring the top band on the operating
    threshold makes the label and the decision the same boundary.
    """
    threshold = trained["metrics"]["test_set_alternative_threshold"]["threshold"]
    p = trained["pipeline"].predict_proba(trained["X_test"])[:, 1]
    for v in p:
        flagged = float(v) >= threshold
        assert (risk_band(float(v), threshold) == "high") == flagged


@_param
def test_risk_bands_in_the_model_card_use_this_models_threshold(trained):
    card_bands = trained["metadata"]["risk_bands"]
    threshold = trained["metrics"]["test_set_alternative_threshold"]["threshold"]
    assert card_bands["operating_threshold"] == pytest.approx(threshold)
    high = next(b for b in card_bands["bands"] if b["label"] == "high")
    assert high["min"] == pytest.approx(threshold)


@_param
def test_pipeline_handles_missing_values(trained):
    """Imputation lives in the pipeline, so a row with NaNs must still score."""
    X = trained["X_test"].copy()
    X.iloc[0, 0] = np.nan
    proba = trained["pipeline"].predict_proba(X.head(1))[:, 1]
    assert 0.0 <= float(proba[0]) <= 1.0


# --- the reported numbers are the model's own ------------------------------------

@_param
def test_reported_metrics_match_a_fresh_evaluation(trained):
    """Re-score the saved pipeline and compare against the published metrics."""
    y_prob = trained["pipeline"].predict_proba(trained["X_test"])[:, 1]
    fresh = classification_metrics(trained["y_test"], y_prob, threshold=0.5)
    claimed = trained["metrics"]["test_set_threshold_0.5"]
    for key in ("roc_auc", "pr_auc", "recall_sensitivity", "specificity",
                "precision", "f1", "accuracy", "brier", "n"):
        assert fresh[key] == pytest.approx(claimed[key], abs=1e-9), (
            f"{trained['disease']} {key}: reported {claimed[key]} but the saved model "
            f"produces {fresh[key]}"
        )


@_param
def test_metric_values_are_in_range(trained):
    m = trained["metrics"]["test_set_threshold_0.5"]
    for key in ("roc_auc", "pr_auc", "recall_sensitivity", "specificity",
                "precision", "f1", "accuracy"):
        assert 0.0 <= m[key] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0
    assert m["n"] == len(trained["y_test"])


@_param
def test_confusion_counts_sum_to_test_size(trained):
    m = trained["metrics"]["test_set_threshold_0.5"]
    assert m["cm_tn"] + m["cm_fp"] + m["cm_fn"] + m["cm_tp"] == len(trained["y_test"])


@_param
def test_alternative_threshold_was_not_chosen_on_the_test_set(trained):
    alt = trained["metrics"]["test_set_alternative_threshold"]
    assert "TRAINING" in alt["threshold_rule"]
    assert 0.0 <= alt["threshold"] <= 1.0


# --- SHAP explains the model it claims to explain --------------------------------

@_param
def test_shap_reconstructs_the_model_output(trained):
    explainer = DiseaseExplainer(
        trained["base_pipeline"], trained["spec"], trained["X_train"], max_background=100
    )
    err = explainer.additivity_error(trained["X_test"].head(25))
    assert err is not None
    assert err < 1e-3, f"SHAP does not reconstruct the model output (max error {err})"


@_param
def test_shap_global_importance_covers_original_features(trained):
    saved = json.loads(
        (REPORTS_DIR / trained["disease"] / "shap_global.json").read_text(encoding="utf-8")
    )
    features = {row["feature"] for row in saved["global_importance"]}
    assert features == set(trained["spec"].features)
    assert all(row["mean_abs_shap"] >= 0 for row in saved["global_importance"])
    assert saved["additivity_max_error"] < 1e-3


@_param
def test_local_explanation_is_complete(trained):
    explainer = DiseaseExplainer(
        trained["base_pipeline"], trained["spec"], trained["X_train"], max_background=100
    )
    exp = explainer.explain_one(trained["X_test"].iloc[[0]])
    assert 0.0 <= exp.predicted_probability <= 1.0
    assert len(exp.contributions) == len(trained["spec"].features)
    # Sorted by absolute contribution, largest first.
    mags = [abs(c["contribution"]) for c in exp.contributions]
    assert mags == sorted(mags, reverse=True)


# --- the model card says what it must ---------------------------------------------

@_param
def test_model_card_is_complete_and_honest(trained):
    card = trained["metadata"]
    for key in ("module", "disease", "disclaimer", "intended_use", "out_of_scope_use",
                "dataset", "protocol", "model", "performance", "explainability",
                "limitations"):
        assert key in card, f"model card missing '{key}'"
    assert "not be interpreted as medical diagnosis" in card["disclaimer"]
    assert card["limitations"], "a model card with no stated limitations is not honest"
    assert card["out_of_scope_use"]
    assert "diagnosis" not in card["module"].lower()
    assert card["model"]["algorithm"]
    assert card["performance"]["test_set_threshold_0.5"]["n"] > 0


@_param
def test_model_card_metrics_match_metrics_json(trained):
    card_metrics = trained["metadata"]["performance"]["test_set_threshold_0.5"]
    report_metrics = trained["metrics"]["test_set_threshold_0.5"]
    assert card_metrics == report_metrics

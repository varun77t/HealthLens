"""Prediction endpoints — one per disease, deliberately not one generic endpoint.

The three modules are independent: different inputs, different models, different
limitations. Three explicit routes keep the OpenAPI schema honest about that (each shows
its own fields and ranges) and make it impossible to submit heart features to the kidney
model. There is no combined endpoint and no aggregate health score.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.schemas.features import DiabetesFeatures, HeartFeatures, KidneyFeatures
from backend.schemas.responses import PredictionResponse
from backend.services import prediction_service as svc
from backend.services.registry import Registry
from backend.routes.deps import get_registry

router = APIRouter(prefix="/predict", tags=["predict"])

_EXPLAIN_QUERY = Query(
    True,
    description=(
        "Include the SHAP explanation. Kidney explanations use SHAP's model-agnostic "
        "KernelExplainer (the selected model is an SVC) and cost ~0.8 s per call, measured; "
        "heart and diabetes use TreeExplainer at ~45 ms. Set false to skip."
    ),
)


def _run(registry: Registry, disease: str, features, include_explanation: bool) -> dict:
    bundle = registry.get(disease)
    values = features.model_dump()
    if svc.is_empty_case(values):
        raise HTTPException(status_code=422, detail=svc.EMPTY_CASE_DETAIL)
    return svc.predict(bundle, values, include_explanation=include_explanation)


@router.post("/heart", response_model=PredictionResponse,
             summary="Heart Disease Presence Prediction")
def predict_heart(
    features: HeartFeatures,
    include_explanation: bool = _EXPLAIN_QUERY,
    registry: Registry = Depends(get_registry),
):
    return _run(registry, "heart", features, include_explanation)


@router.post("/kidney", response_model=PredictionResponse,
             summary="Chronic Kidney Disease Presence Prediction")
def predict_kidney(
    features: KidneyFeatures,
    include_explanation: bool = _EXPLAIN_QUERY,
    registry: Registry = Depends(get_registry),
):
    return _run(registry, "kidney", features, include_explanation)


@router.post("/diabetes", response_model=PredictionResponse,
             summary="Diabetes Health-Indicator Risk Prediction")
def predict_diabetes(
    features: DiabetesFeatures,
    include_explanation: bool = _EXPLAIN_QUERY,
    registry: Registry = Depends(get_registry),
):
    """Estimates a risk *association* with self-reported diabetes or prediabetes status.

    Not a diagnosis and not a forecast: the training data is cross-sectional survey data,
    so a high score means "resembles respondents who reported diabetes", never "will
    develop diabetes".
    """
    return _run(registry, "diabetes", features, include_explanation)

"""Model cards and input schemas.

``GET /models`` is the endpoint a frontend should read before showing any prediction: it
carries each module's limitations and external-validation status alongside its headline
metric, so the caveats travel with the capability rather than living in a separate document
nobody opens.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.routes.deps import get_bundle, get_registry
from backend.schemas.responses import ModelSummary
from backend.services import prediction_service as svc
from backend.services.registry import Registry
from config import DISCLAIMER

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelSummary],
            summary="Model card summary for every loaded module")
def list_models(registry: Registry = Depends(get_registry)):
    return [svc.model_summary(registry.get(d)) for d in registry.diseases]


@router.get("/models/{disease}", summary="The full model card for one module")
def get_model(disease: str, registry: Registry = Depends(get_registry)):
    """The complete ``models/<disease>/metadata.json``, served verbatim."""
    return get_bundle(registry, disease).metadata


@router.get("/models/{disease}/schema",
            summary="Input feature dictionary: ranges, categories and meanings")
def get_schema(disease: str, registry: Registry = Depends(get_registry)):
    """What ``POST /predict/{disease}`` accepts, with training-measured bounds.

    ``observed_min``/``observed_max`` are the range the model was actually fitted on;
    ``hard_min``/``hard_max`` are a mechanical envelope used for validation and encode no
    clinical knowledge. Values between the two are accepted and flagged as extrapolation.
    """
    bundle = get_bundle(registry, disease)
    s = bundle.serving
    return {
        "disease": disease,
        "module": s["module"],
        "feature_order": s["feature_order"],
        "range_note": s["range_note"],
        "dataset_notes": s["spec_notes"],
        "features": s["features"],
        "disclaimer": DISCLAIMER,
    }

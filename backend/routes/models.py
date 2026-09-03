"""Model cards and input schemas.

``GET /models`` is the endpoint a frontend should read before showing any prediction: it
carries each module's limitations and external-validation status alongside its headline
metric, so the caveats travel with the capability rather than living in a separate document
nobody opens.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from backend.routes.deps import get_bundle, get_registry
from backend.schemas.responses import ModelSummary
from backend.services import prediction_service as svc
from backend.services.registry import Registry
from config import DISCLAIMER, MODELS_DIR

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
        "positive_class_meaning": s["positive_class_meaning"],
        "feature_order": s["feature_order"],
        "groups": s.get("groups", []),
        "range_note": s["range_note"],
        "tier_note": s.get("tier_note", ""),
        "value_label_source": s.get("value_label_source", ""),
        "dataset_notes": s["spec_notes"],
        "features": s["features"],
        "disclaimer": DISCLAIMER,
    }


@router.get("/samples/{disease}", tags=["samples"],
            summary="Worked example cases drawn from the held-out test split")
def get_samples(disease: str, registry: Registry = Depends(get_registry)):
    """Real rows from the public dataset, so the app can demonstrate itself with no input.

    These come from the **test** split — records the model was never fitted on — and each
    carries the outcome recorded in the source dataset. That makes a sample a genuine check
    rather than a rehearsal, but it is still one row: the payload's `note` says so, and the
    UI is expected to show it.
    """
    get_bundle(registry, disease)  # 404s on an unknown disease before touching the disk
    path = MODELS_DIR / disease / "samples.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No sample cases exported for '{disease}'. Run "
                   f"`python -m scripts.export_serving_assets`.",
        )
    return {**json.loads(path.read_text(encoding="utf-8")), "disclaimer": DISCLAIMER}

"""Liveness and readiness.

``status`` is ``ok`` only when every configured disease loaded with a working explainer.
A module that loaded but cannot explain reports ``degraded`` rather than ``ok``: this is an
explainability platform, so losing explanations is a real loss of function, not a detail.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.routes.deps import get_registry
from backend.schemas.responses import HealthResponse
from backend.services.registry import Registry
from config import DISCLAIMER, DISEASES

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service and model status")
def health(registry: Registry = Depends(get_registry)):
    status_by_disease = registry.status()
    all_loaded = len(registry.bundles) == len(DISEASES) and not registry.errors
    all_explaining = all(
        s.get("explainer_available") for s in status_by_disease.values() if s.get("loaded")
    )
    if all_loaded and all_explaining:
        status = "ok"
    elif registry.bundles:
        status = "degraded"
    else:
        status = "unavailable"
    return {"status": status, "diseases": status_by_disease, "disclaimer": DISCLAIMER}

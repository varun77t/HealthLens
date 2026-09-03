"""Shared dependencies.

The registry is created once in the app lifespan and stashed on ``app.state``; routes take
it through :func:`get_registry` so tests can substitute a registry loaded without explainers
(which is 5 s faster) without touching the routes.
"""
from __future__ import annotations

from fastapi import HTTPException, Request

from backend.services.registry import ModelBundle, Registry


def get_registry(request: Request) -> Registry:
    registry: Registry | None = getattr(request.app.state, "registry", None)
    if registry is None:  # pragma: no cover - lifespan always sets it
        raise HTTPException(status_code=503, detail="Model registry is not loaded.")
    return registry


def get_bundle(registry: Registry, disease: str) -> ModelBundle:
    """Resolve a disease from a path parameter, with a 404 that names the alternatives."""
    try:
        return registry.get(disease)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"{exc}. Available modules: {', '.join(registry.diseases) or 'none'}.",
        ) from exc

"""Saved analyses — the only write path for health information in this application.

Two rules govern this file.

**The server produces the number it stores.** ``POST /analyses`` takes features, re-runs the
model, and saves its own output. It does not accept a probability. A history assembled from
what the browser claimed would be a table of unprovenanced figures presented to someone as
their own health record.

**Ownership is checked on every read.** Every lookup goes through
:func:`backend.services.analysis_store.get_owned`, which filters on the user, and a row
belonging to someone else comes back as a 404 rather than a 403 — a 403 would confirm the
id exists.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import ValidationError
from sqlalchemy.orm import Session as DbSession

from backend.db import get_db
from backend.routes.deps import get_bundle, get_registry, require_user
from backend.schemas.analyses import (
    AnalysisDetail,
    AnalysisPage,
    AnalysisSummary,
    SaveAnalysisRequest,
    UpdateAnalysisRequest,
)
from backend.schemas.features import DiabetesFeatures, HeartFeatures, KidneyFeatures
from backend.services import analysis_store
from backend.services import prediction_service as svc
from backend.services.registry import Registry
from backend.tables import User

router = APIRouter(prefix="/analyses", tags=["history"])

FEATURE_MODELS = {
    "heart": HeartFeatures,
    "kidney": KidneyFeatures,
    "diabetes": DiabetesFeatures,
}

LIST_NOTE = (
    "Each entry stores the threshold and model version that produced it. A result is never "
    "re-scored against a later model: if the model has changed since, the entry says so and "
    "you can run a new assessment."
)


def _live_version(registry: Registry, disease: str) -> str | None:
    try:
        return registry.get(disease).version
    except KeyError:  # pragma: no cover - a module that failed to load
        return None


@router.post(
    "",
    response_model=AnalysisDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Save an assessment to your history",
)
def save_analysis(
    body: SaveAnalysisRequest,
    user: User = Depends(require_user),
    registry: Registry = Depends(get_registry),
    db: DbSession = Depends(get_db),
):
    bundle = get_bundle(registry, body.disease)

    # Validated against the disease's own schema, exactly as POST /predict/{disease} would.
    # Anything that could not be predicted on must not be storable either.
    try:
        features = FEATURE_MODELS[body.disease](**body.features).model_dump()
    except ValidationError as exc:
        # `ctx` can hold the exception object itself, which json.dumps cannot serialise.
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc

    if svc.is_empty_case(features):
        raise HTTPException(status_code=422, detail=svc.EMPTY_CASE_DETAIL)

    prediction = svc.predict(bundle, features, include_explanation=True)
    row = analysis_store.create(
        db,
        user,
        disease=body.disease,
        features=features,
        prediction=prediction,
        source_kind=body.source_kind,
        source_document=body.source_document,
        label=body.label,
    )
    return AnalysisDetail.of(row, current_version=bundle.version)


@router.get("", response_model=AnalysisPage, summary="Your saved analyses, newest first")
def list_analyses(
    disease: str | None = Query(None, description="Filter to one module."),
    limit: int = Query(analysis_store.DEFAULT_PAGE, ge=1, le=analysis_store.MAX_PAGE),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_user),
    registry: Registry = Depends(get_registry),
    db: DbSession = Depends(get_db),
):
    if disease is not None and disease not in FEATURE_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown module '{disease}'. Available: {', '.join(FEATURE_MODELS)}.",
        )
    rows, total = analysis_store.list_for_user(
        db, user, disease=disease, limit=limit, offset=offset
    )
    versions = {d: _live_version(registry, d) for d in {r.disease for r in rows}}
    return AnalysisPage(
        total=total,
        limit=limit,
        offset=offset,
        items=[AnalysisSummary.of(r, current_version=versions[r.disease]) for r in rows],
        note=LIST_NOTE,
    )


@router.get("/{analysis_id}", response_model=AnalysisDetail, summary="One saved analysis")
def get_analysis(
    analysis_id: str,
    user: User = Depends(require_user),
    registry: Registry = Depends(get_registry),
    db: DbSession = Depends(get_db),
):
    row = analysis_store.get_owned(db, user, analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No saved analysis with that id.")
    return AnalysisDetail.of(row, current_version=_live_version(registry, row.disease))


@router.patch("/{analysis_id}", response_model=AnalysisDetail, summary="Rename a saved analysis")
def update_analysis(
    analysis_id: str,
    body: UpdateAnalysisRequest,
    user: User = Depends(require_user),
    registry: Registry = Depends(get_registry),
    db: DbSession = Depends(get_db),
):
    row = analysis_store.get_owned(db, user, analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No saved analysis with that id.")
    row = analysis_store.set_label(db, row, body.label)
    return AnalysisDetail.of(row, current_version=_live_version(registry, row.disease))


@router.delete(
    "/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a saved analysis",
)
def delete_analysis(
    analysis_id: str,
    user: User = Depends(require_user),
    db: DbSession = Depends(get_db),
):
    row = analysis_store.get_owned(db, user, analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No saved analysis with that id.")
    analysis_store.delete(db, row)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

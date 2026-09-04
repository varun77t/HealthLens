"""Reading and writing saved analyses. Scoped to one user, always.

Every function here takes the owning :class:`~backend.tables.User` and filters on it. There
is no "get by id" that does not also check ownership: an unscoped lookup is one forgotten
``if`` away from serving one person's health record to another, and the way to make that
impossible is to have no function that can do it.

A row that belongs to someone else is reported as **absent**, not forbidden. A 403 confirms
the id exists, which turns the endpoint into an oracle for enumerating other people's
records.
"""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from backend.tables import Analysis, User

MAX_PAGE = 100
DEFAULT_PAGE = 20


def create(
    db: DbSession,
    user: User,
    *,
    disease: str,
    features: dict,
    prediction: dict,
    source_kind: str,
    source_document: str | None,
    label: str | None,
) -> Analysis:
    """Store a result **the server produced**.

    ``prediction`` is the output of :func:`backend.services.prediction_service.predict`,
    re-run by the route from these very features. Nothing here comes from the client's idea
    of what the model said: a saved history of numbers the browser asserted would be a
    record of nothing.
    """
    band = prediction["risk_band"]
    row = Analysis(
        user_id=user.id,
        disease=disease,
        features=features,
        source_kind=source_kind,
        source_document=(source_document or None),
        label=(label or "").strip() or None,
        probability=prediction["probability"],
        flagged=prediction["flagged"],
        threshold=prediction["threshold"],
        band_label=band["label"],
        band_lower=band["lower"],
        band_upper=band["upper"],
        model_name=prediction["model_name"],
        model_version=prediction["model_version"],
        calibration=prediction["calibration"],
        imputed_features=prediction["imputed_features"],
        explanation=prediction.get("explanation"),
        warnings=prediction["warnings"],
        note=prediction["disclaimer"],
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_for_user(
    db: DbSession,
    user: User,
    *,
    disease: str | None = None,
    limit: int = DEFAULT_PAGE,
    offset: int = 0,
) -> tuple[list[Analysis], int]:
    limit = max(1, min(limit, MAX_PAGE))
    where = [Analysis.user_id == user.id]
    if disease:
        where.append(Analysis.disease == disease)

    total = int(
        db.execute(select(func.count()).select_from(Analysis).where(*where)).scalar_one()
    )
    rows = (
        db.execute(
            select(Analysis)
            .where(*where)
            # `id` breaks ties: two saves in the same instant would otherwise page
            # non-deterministically and could show or skip a row.
            .order_by(Analysis.created_at.desc(), Analysis.id.desc())
            .limit(limit)
            .offset(max(0, offset))
        )
        .scalars()
        .all()
    )
    return list(rows), total


def get_owned(db: DbSession, user: User, analysis_id: str) -> Analysis | None:
    """The row, only if this user owns it. Someone else's id returns ``None``."""
    return db.execute(
        select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user.id)
    ).scalar_one_or_none()


def set_label(db: DbSession, analysis: Analysis, label: str | None) -> Analysis:
    """Rename a saved analysis. The only field a saved row will ever accept a change to.

    The inputs and the result are immutable: editing them would leave a record whose
    probability was never produced from the values shown beside it. Changing an answer means
    running a new assessment, which is the same rule the review screen already enforces.
    """
    analysis.label = (label or "").strip() or None
    db.commit()
    db.refresh(analysis)
    return analysis


def delete(db: DbSession, analysis: Analysis) -> None:
    db.delete(analysis)
    db.commit()

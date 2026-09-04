"""Request and response bodies for saved analyses.

The save request carries **inputs only**. There is deliberately no field for a probability,
a risk band or a model version: the server re-runs the model from the features and stores
what it produced. A client-supplied number would be a figure with no provenance sitting in
someone's health record, which is the one thing this project has refused to ship since
Phase 1.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.tables import Analysis, as_utc
from config import DISEASES

DiseaseName = Literal["heart", "kidney", "diabetes"]
SourceKind = Literal["upload", "manual", "sample"]


class SaveAnalysisRequest(BaseModel):
    disease: DiseaseName
    features: dict[str, float | None] = Field(
        description="The same body POST /predict/{disease} takes. Validated against that "
        "model's own schema before anything is stored."
    )
    source_kind: SourceKind = "manual"
    source_document: str | None = Field(
        default=None,
        max_length=255,
        description="Filename only, for provenance on the saved report. The document "
        "itself is never stored.",
    )
    label: str | None = Field(
        default=None, max_length=120, description="Your own note, e.g. 'before diet change'."
    )


class UpdateAnalysisRequest(BaseModel):
    label: str | None = Field(default=None, max_length=120)


class AnalysisSummary(BaseModel):
    """The list row. Carries the threshold, because the band is meaningless without it."""

    id: str
    disease: DiseaseName
    module: str
    created_at: datetime
    label: str | None
    source_kind: SourceKind
    source_document: str | None
    probability: float
    flagged: bool
    threshold: float
    band_label: str
    model_name: str
    model_version: str
    #: False when the live model is no longer the one that produced this result.
    model_is_current: bool

    @classmethod
    def of(cls, row: Analysis, *, current_version: str | None) -> "AnalysisSummary":
        return cls(
            id=row.id,
            disease=row.disease,  # type: ignore[arg-type]
            module=DISEASES[row.disease]["label"],
            created_at=as_utc(row.created_at),
            label=row.label,
            source_kind=row.source_kind,  # type: ignore[arg-type]
            source_document=row.source_document,
            probability=row.probability,
            flagged=row.flagged,
            threshold=row.threshold,
            band_label=row.band_label,
            model_name=row.model_name,
            model_version=row.model_version,
            model_is_current=(current_version is None or current_version == row.model_version),
        )


class AnalysisDetail(AnalysisSummary):
    features: dict[str, float | None]
    band_lower: float
    band_upper: float
    calibration: str
    imputed_features: list[str]
    explanation: dict | None
    warnings: list[str]
    #: The version running now, when it differs from the one that produced this result.
    current_model_version: str | None
    disclaimer: str

    @classmethod
    def of(cls, row: Analysis, *, current_version: str | None) -> "AnalysisDetail":
        base = AnalysisSummary.of(row, current_version=current_version)
        return cls(
            **base.model_dump(),
            features=row.features,
            band_lower=row.band_lower,
            band_upper=row.band_upper,
            calibration=row.calibration,
            imputed_features=list(row.imputed_features or []),
            explanation=row.explanation,
            warnings=list(row.warnings or []),
            current_model_version=(
                current_version if current_version != row.model_version else None
            ),
            disclaimer=row.note or "",
        )


class AnalysisPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[AnalysisSummary]
    note: str

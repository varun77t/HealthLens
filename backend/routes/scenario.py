"""Scenario endpoint — re-score a case with changed inputs.

This is the endpoint most likely to be misread, so the guardrails are part of the contract:

* Both estimates are returned side by side. There is no single "improved risk" number.
* Every response leads with the illustrative-scenario label.
* Overriding a feature that is plausibly a *consequence* of the disease (or is not
  modifiable at all — age, sex) adds a warning naming that feature. The models are
  cross-sectional associations; nothing here supports a causal or counterfactual claim.

Both the base case and the overrides are validated against the same schema as ``/predict``,
so a scenario cannot smuggle in a value the prediction endpoint would reject.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError

from backend.routes.deps import get_bundle, get_registry
from backend.schemas.features import FEATURE_MODELS
from backend.schemas.responses import ScenarioRequest, ScenarioResponse
from backend.services import prediction_service as svc
from backend.services.registry import Registry

router = APIRouter(prefix="/scenario", tags=["scenario"])


def _validated(disease: str, values: dict, *, what: str) -> dict:
    model = FEATURE_MODELS[disease]
    try:
        return model(**values).model_dump()
    except ValidationError as exc:
        # `exc.errors()` embeds the original exception object in `ctx`, which will not
        # serialise; `exc.json()` renders the same errors as JSON-safe values.
        raise HTTPException(
            status_code=422, detail={what: json.loads(exc.json())}
        ) from exc


@router.post("/{disease}", response_model=ScenarioResponse,
             summary="Illustrative model scenario: the same case with changed inputs")
def scenario(
    disease: str,
    payload: ScenarioRequest,
    include_explanation: bool = Query(
        False,
        description="Include SHAP for both estimates. Off by default: a scenario doubles "
                    "the explainer cost, which is ~0.8 s per call for kidney.",
    ),
    registry: Registry = Depends(get_registry),
):
    bundle = get_bundle(registry, disease)
    base = _validated(disease, payload.features, what="features")
    if all(v is None for v in base.values()):
        raise HTTPException(
            status_code=422,
            detail="No features supplied in the base case; there is nothing to vary.",
        )
    if not payload.overrides:
        raise HTTPException(
            status_code=422,
            detail="No overrides supplied. Use POST /predict/{disease} to score one case.",
        )
    # Overrides are validated on their own so an out-of-range override is reported as an
    # override error rather than being merged in and blamed on the base case.
    # (`extra="forbid"` on the feature model means an unknown override name is rejected
    #  here too, so a typo cannot be silently ignored and reported as "no change".)
    _validated(disease, payload.overrides, what="overrides")
    return svc.scenario(
        bundle, base, dict(payload.overrides), include_explanation=include_explanation
    )

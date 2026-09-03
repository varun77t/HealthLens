"""Request schemas — one per disease, built from that disease's exported feature dictionary.

The three diseases have entirely different inputs (13 clinical measurements, 24 laboratory
values, 21 survey answers), so they get three separate schemas and share nothing but the
construction code.

Those schemas are **generated** from ``models/<disease>/serving.json`` rather than typed out
by hand. Fifty-eight fields with individual bounds and category lists is exactly the kind of
thing that acquires silent transcription errors, and a bound that disagrees with the training
data is worse than no bound at all — it would reject valid inputs or admit ones the encoder
cannot represent. Generating them means the API can only advertise ranges that were measured.

Validation policy, in three levels:

* **Reject** a value outside the mechanical envelope in ``serving.json`` (one observed range
  either side of the training minimum/maximum). This catches unit mix-ups and typos.
* **Reject** an unrecognised category. The one-hot encoder runs with
  ``handle_unknown="ignore"``, so an unknown level is encoded as all-zeros and yields a
  confident-looking prediction for a case whose category the model never saw. Silently
  accepting that is the failure mode worth spending an error on.
* **Accept and flag** a value inside the envelope but outside the observed training range.
  The model has no training support there; that is a caveat for the response, not an error.

Every field is optional. The pipeline's imputers were fitted on the training folds, so a
missing value is handled the way the training data's missing values were — but a request that
omits most fields is scored almost entirely from imputed medians, so the response reports
exactly which features were imputed.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from config import MODELS_DIR


@lru_cache(maxsize=None)
def serving_spec(disease: str) -> dict:
    """The exported serving descriptor for ``disease`` (cached; read once per process)."""
    path = MODELS_DIR / disease / "serving.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m scripts.export_serving_assets` after "
            f"training to generate the serving assets for '{disease}'."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _describe(feature: dict) -> str:
    base = feature["description"]
    if feature["kind"] == "numeric":
        return (
            f"{base}. Observed in training: {feature['observed_min']:g} to "
            f"{feature['observed_max']:g} (median {feature['observed_median']:g}). "
            f"Accepted range {feature['hard_min']:g} to {feature['hard_max']:g}; "
            "values outside the observed range are accepted but flagged as extrapolation."
        )
    cats = ", ".join(f"{c:g}" for c in feature["categories"])
    return f"{base}. Allowed values: {cats}. Omit or send null to have it imputed."


def _field(feature: dict):
    """A pydantic field carrying this feature's measured bounds."""
    kwargs: dict[str, Any] = {"default": None, "description": _describe(feature)}
    if feature["kind"] == "numeric":
        kwargs["ge"] = feature["hard_min"]
        kwargs["le"] = feature["hard_max"]
    else:
        kwargs["json_schema_extra"] = {"enum": list(feature["categories"])}
    return (float | None, Field(**kwargs))


def build_feature_model(disease: str) -> type[BaseModel]:
    """Create the request model for ``disease`` from its exported feature dictionary."""
    spec = serving_spec(disease)
    fields = {f["name"]: _field(f) for f in spec["features"]}
    allowed = {
        f["name"]: set(f["categories"])
        for f in spec["features"]
        if f["kind"] in ("categorical", "binary")
    }

    def check_categories(self):
        bad = []
        for name, options in allowed.items():
            value = getattr(self, name, None)
            if value is not None and float(value) not in options:
                shown = ", ".join(f"{c:g}" for c in sorted(options))
                bad.append(f"'{name}' must be one of [{shown}], got {value:g}")
        if bad:
            # Not a warning: an unknown category is one-hot encoded as all zeros, which
            # would silently score the case as if the category had never been asked about.
            raise ValueError("; ".join(bad))
        return self

    model = create_model(
        f"{disease.capitalize()}Features",
        __config__=ConfigDict(
            extra="forbid",
            json_schema_extra={
                "description": (
                    f"Input features for the {spec['module']} model. "
                    "All fields optional; omitted fields are imputed by the pipeline's "
                    "training-fitted imputer and reported back in `imputed_features`."
                )
            },
        ),
        __validators__={
            "_check_categories": model_validator(mode="after")(check_categories)
        },
        **fields,
    )
    model.__doc__ = (
        f"Input features for {spec['module']} (bounds measured on the training split)."
    )
    return model


def example_payload(disease: str) -> dict:
    """A syntactically valid example: the training median / most common level per feature.

    Used for OpenAPI examples and smoke tests. It is a synthetic profile assembled from
    per-feature training statistics, not a real patient and not a case with a known label.
    """
    spec = serving_spec(disease)
    out = {}
    for f in spec["features"]:
        if f["kind"] == "numeric":
            out[f["name"]] = f["observed_median"]
        else:
            out[f["name"]] = f["categories"][0]
    return out


HeartFeatures = build_feature_model("heart")
KidneyFeatures = build_feature_model("kidney")
DiabetesFeatures = build_feature_model("diabetes")

FEATURE_MODELS: dict[str, type[BaseModel]] = {
    "heart": HeartFeatures,
    "kidney": KidneyFeatures,
    "diabetes": DiabetesFeatures,
}

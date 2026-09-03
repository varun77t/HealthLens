"""Model registry — every artifact loaded once, at startup.

Nothing in the request path fits, trains, resamples or reads a dataset. The registry holds,
per disease: the calibrated serving pipeline, the uncalibrated base pipeline SHAP needs, the
model card, the serving descriptor, and a ready-built ``DiseaseExplainer``.

Building the explainers at startup is deliberate. The kidney model is an SVC, so its
explainer is SHAP's model-agnostic ``KernelExplainer``, which costs 4.7 s to construct
(measured; see ``models/kidney/serving.json``). Doing that lazily would move the cost onto
whichever unlucky request arrived first.

If an explainer fails to build, the service still serves predictions but records the failure
and reports ``explanation: null`` with the reason. It is never silently downgraded — an
explainability platform that quietly stops explaining is worse than one that says it broke.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import joblib

from config import DISEASES, MODELS_DIR
from ml.data.feature_spec import FeatureSpec


@dataclass
class ModelBundle:
    disease: str
    pipeline: Any                # calibrated — produces the probability
    base_pipeline: Any           # uncalibrated — what SHAP explains
    metadata: dict               # models/<disease>/metadata.json
    serving: dict                # models/<disease>/serving.json
    explainer: Any | None = None
    explainer_error: str | None = None
    load_seconds: float = 0.0
    features: dict[str, dict] = field(default_factory=dict)

    @property
    def feature_order(self) -> list[str]:
        return list(self.serving["feature_order"])

    @property
    def threshold(self) -> float:
        return float(self.serving["operating_threshold"])

    @property
    def version(self) -> str:
        return str(self.metadata.get("created", "unknown"))


def _spec_from_serving(serving: dict) -> FeatureSpec:
    """Rebuild the FeatureSpec the explainer needs, without loading the dataset."""
    kinds: dict[str, list[str]] = {"numeric": [], "categorical": [], "binary": []}
    descriptions = {}
    for f in serving["features"]:
        kinds[f["kind"]].append(f["name"])
        descriptions[f["name"]] = f["description"]
    return FeatureSpec(
        disease=serving["disease"],
        target=serving.get("target", serving["disease"]),
        numeric=kinds["numeric"],
        categorical=kinds["categorical"],
        binary=kinds["binary"],
        descriptions=descriptions,
        notes=serving.get("spec_notes", ""),
    )


def load_bundle(disease: str, *, with_explainer: bool = True) -> ModelBundle:
    """Load every artifact for one disease. Raises if a required file is missing."""
    t0 = time.perf_counter()
    d = MODELS_DIR / disease
    required = ["pipeline.joblib", "base_pipeline.joblib", "metadata.json", "serving.json"]
    missing = [name for name in required if not (d / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Model artifacts missing for '{disease}': {', '.join(missing)}. "
            f"Run `python scripts/train_{disease}.py` then "
            f"`python -m scripts.export_serving_assets`."
        )

    serving = json.loads((d / "serving.json").read_text(encoding="utf-8"))
    bundle = ModelBundle(
        disease=disease,
        pipeline=joblib.load(d / "pipeline.joblib"),
        base_pipeline=joblib.load(d / "base_pipeline.joblib"),
        metadata=json.loads((d / "metadata.json").read_text(encoding="utf-8")),
        serving=serving,
        features={f["name"]: f for f in serving["features"]},
    )

    if with_explainer:
        try:
            from ml.explainability.shap_explainer import DiseaseExplainer

            background = joblib.load(d / "shap_background.joblib")
            bundle.explainer = DiseaseExplainer(
                bundle.base_pipeline,
                _spec_from_serving(serving),
                background,
                max_background=len(background),
            )
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            bundle.explainer_error = f"{type(exc).__name__}: {exc}"

    bundle.load_seconds = round(time.perf_counter() - t0, 3)
    return bundle


class Registry:
    """All disease bundles, keyed by disease name."""

    def __init__(self, bundles: dict[str, ModelBundle], errors: dict[str, str]):
        self.bundles = bundles
        self.errors = errors

    def __contains__(self, disease: str) -> bool:
        return disease in self.bundles

    def get(self, disease: str) -> ModelBundle:
        if disease not in self.bundles:
            reason = self.errors.get(disease, "not a known disease module")
            raise KeyError(f"No model available for '{disease}': {reason}")
        return self.bundles[disease]

    @property
    def diseases(self) -> list[str]:
        return list(self.bundles)

    def status(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, b in self.bundles.items():
            out[name] = {
                "loaded": True,
                "model": b.serving["model"],
                "calibration": b.serving["calibration"],
                "version": b.version,
                "operating_threshold": b.threshold,
                "explainer": (b.serving.get("explainer") or {}).get("kind"),
                "explainer_available": b.explainer is not None,
                "explainer_error": b.explainer_error,
                "load_seconds": b.load_seconds,
            }
        for name, err in self.errors.items():
            out[name] = {"loaded": False, "error": err}
        return out


def load_registry(*, with_explainer: bool = True) -> Registry:
    """Load every disease that has artifacts; record, but do not raise on, the rest."""
    bundles: dict[str, ModelBundle] = {}
    errors: dict[str, str] = {}
    for disease in DISEASES:
        try:
            bundles[disease] = load_bundle(disease, with_explainer=with_explainer)
        except Exception as exc:  # noqa: BLE001 - surfaced through /health
            errors[disease] = f"{type(exc).__name__}: {exc}"
    return Registry(bundles, errors)

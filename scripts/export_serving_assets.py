"""Export everything the API needs to serve a disease model, so it never touches `data/`.

The backend must not read the raw datasets. Two things it would otherwise need them for:

1. **Input validation ranges.** A request schema needs to know that `age` is a number and
   that `cp` takes the values {1, 2, 3, 4} — the latter matters because the one-hot encoder
   is configured `handle_unknown="ignore"`, so an unrecognised category is silently encoded
   as all-zeros and produces a confident-looking prediction from a case the model never saw
   a category for. The API rejects unknown categories instead.
2. **A SHAP background sample.** ``DiseaseExplainer`` needs reference rows. Exporting the
   exact 200 rows the training run used (same ``RANDOM_STATE``, same sampling) means the
   served explanation is the one that was validated, not a re-derived approximation.

Every number written here is measured from the training split of the real dataset. Ranges
are taken from the **training** rows only — the test split is not consulted, so nothing the
API advertises was derived from held-out data.

Run: ``python -m scripts.export_serving_assets``
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    DISEASES,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    RISK_BAND_NOTE,
    risk_bands,
)
from ml.data.loaders import load
from ml.reporting import write_json
from ml.training.splits import make_split

BACKGROUND_ROWS = 200

# How far outside the observed training range a value may go before it is rejected
# outright. This is a mechanical envelope (one observed range either side), NOT a
# clinical reference range — nothing here encodes medical knowledge about plausible
# values. Anything outside the observed range but inside the envelope is accepted and
# flagged as extrapolation.
ENVELOPE_MULTIPLE = 1.0


def _numeric_descriptor(col: str, values: pd.Series, spec) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    pad = ENVELOPE_MULTIPLE * span if span > 0 else max(abs(hi), 1.0)
    hard_min = lo - pad
    if lo >= 0:  # a quantity never observed negative is not allowed to go negative
        hard_min = max(0.0, hard_min)
    return {
        "name": col,
        "kind": "numeric",
        "description": spec.describe(col),
        "observed_min": lo,
        "observed_max": hi,
        "observed_median": float(v.median()),
        "hard_min": hard_min,
        "hard_max": hi + pad,
        "n_missing_in_training": int(values.isna().sum()),
    }


def _categorical_descriptor(col: str, values: pd.Series, spec) -> dict:
    cats = sorted(float(c) for c in pd.unique(values.dropna()))
    return {
        "name": col,
        "kind": "categorical",
        "description": spec.describe(col),
        "categories": cats,
        "n_missing_in_training": int(values.isna().sum()),
    }


def _binary_descriptor(col: str, values: pd.Series, spec) -> dict:
    observed = sorted(float(c) for c in pd.unique(values.dropna()))
    return {
        "name": col,
        "kind": "binary",
        "description": spec.describe(col),
        "categories": observed,
        "n_missing_in_training": int(values.isna().sum()),
    }


def _feature_descriptors(X_train: pd.DataFrame, spec) -> list[dict]:
    out = []
    for col in spec.features:
        series = X_train[col]
        if col in spec.numeric:
            out.append(_numeric_descriptor(col, series, spec))
        elif col in spec.categorical:
            out.append(_categorical_descriptor(col, series, spec))
        else:
            out.append(_binary_descriptor(col, series, spec))
    return out


def _measure_explainer(base_pipeline, spec, background: pd.DataFrame, X_probe: pd.DataFrame):
    """Build the explainer and time it, so the API's cost is a measured number."""
    from ml.explainability.shap_explainer import DiseaseExplainer

    t0 = time.perf_counter()
    explainer = DiseaseExplainer(
        base_pipeline, spec, background, max_background=len(background)
    )
    init_s = time.perf_counter() - t0

    timings = []
    for i in range(min(5, len(X_probe))):
        t0 = time.perf_counter()
        explainer.explain_one(X_probe.iloc[[i]])
        timings.append(time.perf_counter() - t0)
    return explainer, {
        "kind": explainer.kind,
        "init_seconds": round(init_s, 3),
        "explain_one_median_ms": round(1000 * float(np.median(timings)), 1),
        "explain_one_max_ms": round(1000 * float(np.max(timings)), 1),
        "n_timed_calls": len(timings),
        "background_rows": int(len(background)),
    }


def _amend_risk_bands(disease: str, threshold: float) -> Path | None:
    """Rewrite the risk-band block of an existing model card without retraining.

    The bands moved from a fixed 0.33/0.66 grid to the model's own operating threshold.
    Rebuilding the whole card would mean retraining; only this block changes, and
    ``_card_markdown`` is a pure function of the card dict, so the markdown can be
    regenerated from the amended JSON.
    """
    from ml.model_card import _card_markdown

    path = MODELS_DIR / disease / "metadata.json"
    if not path.exists():
        return None
    card = json.loads(path.read_text(encoding="utf-8"))
    card["risk_bands"] = {
        "note": RISK_BAND_NOTE,
        "operating_threshold": threshold,
        "bands": [{"label": n, "min": lo, "max": hi} for n, lo, hi in risk_bands(threshold)],
    }
    write_json(path, card)
    from ml.reporting import write_lines

    return write_lines(MODELS_DIR / disease / "MODEL_CARD.md", _card_markdown(card))


def export_disease(disease: str, *, background_rows: int = BACKGROUND_ROWS,
                   write: bool = True) -> dict:
    """Write ``serving.json`` + ``shap_background.joblib`` for one disease."""
    models_dir = MODELS_DIR / disease
    metrics = json.loads(
        (REPORTS_DIR / disease / "metrics.json").read_text(encoding="utf-8")
    )
    alt = metrics["test_set_alternative_threshold"]
    threshold = float(alt["threshold"])

    X, y, spec = load(disease)
    split = make_split(disease, X, y)
    X_train, X_test, y_train, _ = split.apply(X, y)

    # The exact rows the training run's explainer used: same sample size, same seed.
    background = (
        X_train
        if len(X_train) <= background_rows
        else X_train.sample(background_rows, random_state=RANDOM_STATE)
    )

    base_pipeline = joblib.load(models_dir / "base_pipeline.joblib")
    _, explainer_profile = _measure_explainer(base_pipeline, spec, background, X_test)

    payload = {
        "disease": disease,
        "module": DISEASES[disease]["label"],
        "positive_class_meaning": DISEASES[disease]["positive"],
        "generated": date.today().isoformat(),
        "generated_by": "scripts/export_serving_assets.py",
        "model": metrics["model"],
        "calibration": metrics["calibration"],
        "operating_threshold": threshold,
        "threshold_rule": alt["threshold_rule"],
        "threshold_test_recall": alt["recall_sensitivity"],
        "threshold_test_specificity": alt["specificity"],
        "threshold_test_precision": alt["precision"],
        "metrics_at_0.5": {
            k: metrics["test_set_threshold_0.5"][k]
            for k in ("recall_sensitivity", "specificity", "precision", "accuracy")
        },
        "risk_bands": {
            "note": RISK_BAND_NOTE,
            "operating_threshold": threshold,
            "bands": [
                {"label": n, "min": lo, "max": hi} for n, lo, hi in risk_bands(threshold)
            ],
        },
        "training_positive_rate": float(y_train.mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "roc_auc_test": metrics["test_set_threshold_0.5"]["roc_auc"],
        "external_validation": _external_validation_status(disease),
        "explainer": explainer_profile,
        "range_note": (
            "`observed_min`/`observed_max` are the minimum and maximum seen in the "
            f"{len(X_train):,} TRAINING rows. `hard_min`/`hard_max` widen that by one "
            "observed range either side (clipped at zero for quantities never observed "
            "negative). This envelope is mechanical, not clinical: it encodes no medical "
            "knowledge about which values are physiologically possible. Values outside "
            "the observed range but inside the envelope are accepted and flagged as "
            "extrapolation, because the model has no training support there."
        ),
        "features": _feature_descriptors(X_train, spec),
        "feature_order": list(spec.features),
        "spec_notes": spec.notes,
    }

    if write:
        write_json(models_dir / "serving.json", payload)
        joblib.dump(background.reset_index(drop=True), models_dir / "shap_background.joblib")
        _amend_risk_bands(disease, threshold)
    return payload


def _external_validation_status(disease: str) -> dict:
    """Say plainly whether this model was ever checked against a second cohort."""
    path = REPORTS_DIR / disease / "external_validation.json"
    if not path.exists():
        return {
            "status": "never_attempted",
            "summary": (
                "No external validation was attempted. Every performance figure comes "
                "from a single held-out split of one dataset."
            ),
        }
    ext = json.loads(path.read_text(encoding="utf-8"))
    if ext.get("usable_as_external_validation"):
        return {"status": "validated", "summary": ext.get("verdict", ""),
                "dataset": ext.get("external_dataset")}
    return {
        "status": "rejected",
        "dataset": ext.get("external_dataset"),
        "summary": ext.get("verdict", ""),
    }


def main() -> None:
    for disease in DISEASES:
        payload = export_disease(disease)
        ex = payload["explainer"]
        print(
            f"[{disease}] threshold={payload['operating_threshold']:.4f} "
            f"features={len(payload['features'])} "
            f"explainer={ex['kind']} ({ex['explain_one_median_ms']} ms/call) "
            f"external={payload['external_validation']['status']}"
        )


if __name__ == "__main__":
    main()

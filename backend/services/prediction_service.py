"""Turning a validated request into a prediction, an explanation, and its caveats.

No model is fitted here. The pipeline and explainer were built at startup by
:mod:`backend.services.registry`; this module transforms one request into one row, calls
``predict_proba``, and assembles everything a caller needs to read the number correctly.

Most of the code is that last part. Seven findings from Phases 3-7 have to survive into the
API rather than being rediscovered by whoever consumes it:

1. The old fixed risk bands were wrong for diabetes, so bands are anchored on each model's
   own operating threshold (see :data:`config.RISK_BAND_NOTE`).
2. 0.5 is not the diabetes operating point — the model flags at 0.1388 — so the decision
   reported is always taken at the model's own threshold, and 0.5 is never used.
3. Kidney explanations cost ~0.8 s each (KernelExplainer over an SVC, measured). The cost is
   published in ``/models`` and explanations can be switched off per request.
4. No module has external validation, and heart's attempt was rejected. Every prediction
   says so.
5. Diabetes error profiles differ sharply by age band, so where Phase 7 measured a subgroup
   reliably, the response carries that subgroup's real recall and specificity.
6. Inputs outside the training range are flagged rather than silently scored.
7. SHAP explains the uncalibrated base pipeline, not the calibrated probability. Both
   numbers are returned.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from backend.services.registry import ModelBundle
from config import DISCLAIMER, REPORTS_DIR, RISK_BAND_NOTE, risk_bands

SHAP_NOTE = (
    "SHAP contributions explain the UNCALIBRATED base pipeline, whose output is "
    "`uncalibrated_probability`. The `probability` field above comes from the calibrated "
    "wrapper. Calibration is a monotonic transform of the base score, so the sign and the "
    "ranking of these contributions carry over unchanged; their magnitudes do not sum to "
    "the calibrated probability and must not be presented as if they did."
)

SCENARIO_LABEL = (
    "Illustrative model scenario — shows how the trained model responds to changed inputs. "
    "It is not a prediction of real medical risk, and not a statement that changing the "
    "input would change anyone's health. The model is a cross-sectional association fitted "
    "to observational data; it supports no causal claim."
)

# Features that are plausibly consequences of the disease rather than causes of it. Editing
# one in a scenario produces a number, but the number does not mean what a reader will
# assume it means. Hand-authored from each model card's limitations, not generated.
NON_ACTIONABLE: dict[str, dict[str, str]] = {
    "diabetes": {
        "GenHlth": "self-rated general health is plausibly a consequence of diabetes, not a "
                   "lever on it",
        "DiffWalk": "difficulty walking is plausibly a consequence of diabetes",
        "PhysHlth": "days of poor physical health is plausibly a consequence of diabetes",
        "HeartDiseaseorAttack": "heart disease history is a correlate, not a modifiable input",
        "Age": "age is not modifiable; changing it asks how the model scores a different "
               "person, not how this person could change",
        "Sex": "sex is not modifiable; changing it asks about a different person",
    },
    "heart": {
        "thal": "a thallium scan result reflects existing disease",
        "ca": "vessels coloured by fluoroscopy reflects existing disease",
        "oldpeak": "exercise-induced ST depression reflects existing disease",
        "exang": "exercise-induced angina reflects existing disease",
        "age": "age is not modifiable; changing it asks about a different person",
        "sex": "sex is not modifiable; changing it asks about a different person",
    },
    "kidney": {
        "sc": "serum creatinine is a diagnostic criterion for CKD, not an input to change",
        "hemo": "haemoglobin reflects existing disease",
        "pcv": "packed cell volume reflects existing disease",
        "sg": "urine specific gravity is a diagnostic criterion",
        "al": "albuminuria is a diagnostic criterion for CKD",
        "age": "age is not modifiable; changing it asks about a different person",
    },
}


# ------------------------------------------------------------------------------------
# Subgroup performance (Phase 7), attached where it was measured reliably
# ------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _subgroup_table(disease: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / disease / "fairness" / "subgroups.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    return table[table["threshold_name"] == "operating"]


def _subgroup_performance(bundle: ModelBundle, row: pd.DataFrame) -> dict | None:
    """This case's age band, with the recall/specificity Phase 7 actually measured for it.

    Returned only when that subgroup was marked reliable, so the small-n heart and kidney
    subgroups (1 of 6 and 0 of 4 reliable) contribute nothing rather than noise.
    """
    table = _subgroup_table(bundle.disease)
    age_col = next((c for c in row.columns if c.lower() == "age"), None)
    if table is None or age_col is None:
        return None
    value = row.iloc[0][age_col]
    if pd.isna(value):
        return None

    from ml.fairness.subgroup_metrics import age_bands

    band = age_bands(pd.Series([value]), bundle.disease).iloc[0]
    match = table[(table["attribute"] == "age_band") & (table["subgroup"] == band)]
    if match.empty or not bool(match.iloc[0]["reliable"]):
        return None
    r = match.iloc[0]
    return {
        "attribute": "age_band",
        "subgroup": str(band),
        "n_test_rows": int(r["n"]),
        "recall": float(r["recall"]),
        "specificity": float(r["specificity"]),
        "roc_auc": float(r["roc_auc"]),
        "note": (
            f"Measured on the {int(r['n']):,} held-out test rows in the '{band}' age band "
            f"at this same threshold: recall {r['recall']:.3f}, specificity "
            f"{r['specificity']:.3f}. A single global threshold does not behave the same "
            "way in every age band, so these describe this case's band better than the "
            "overall figures do. They are cohort-level rates, not this individual's "
            "probability of being right."
        ),
    }


# ------------------------------------------------------------------------------------
# Request -> row
# ------------------------------------------------------------------------------------


def build_row(bundle: ModelBundle, payload: dict) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """One-row frame in training column order, plus what was missing and what was extreme."""
    imputed: list[str] = []
    extrapolated: list[dict] = []
    values: dict[str, float] = {}

    for name in bundle.feature_order:
        value = payload.get(name)
        if value is None:
            imputed.append(name)
            values[name] = np.nan
            continue
        value = float(value)
        values[name] = value
        f = bundle.features[name]
        if f["kind"] == "numeric" and not (f["observed_min"] <= value <= f["observed_max"]):
            extrapolated.append({
                "feature": name,
                "value": value,
                "observed_min": f["observed_min"],
                "observed_max": f["observed_max"],
                "note": "outside the range seen in training; the model has no support here",
            })

    row = pd.DataFrame([values], columns=bundle.feature_order).astype("float64")
    return row, imputed, extrapolated


def _warnings(bundle: ModelBundle, imputed: list[str], extrapolated: list[dict]) -> list[str]:
    out: list[str] = []
    n_total = len(bundle.feature_order)

    if imputed:
        pct = 100.0 * len(imputed) / n_total
        msg = (
            f"{len(imputed)} of {n_total} features ({pct:.0f}%) were not supplied and were "
            "filled by the pipeline's training-fitted imputer."
        )
        if pct >= 50:
            msg += (
                " More than half the input is imputed, so this score describes the average "
                "training profile more than it describes the case submitted."
            )
        out.append(msg)

    if extrapolated:
        names = ", ".join(e["feature"] for e in extrapolated)
        out.append(
            f"Outside the training range: {names}. The model was never fitted on values "
            "there and its behaviour is unconstrained; treat the output as unreliable."
        )

    ext = bundle.serving.get("external_validation", {})
    if ext.get("status") == "rejected":
        out.append(
            "This model has NO external validation. The intended external cohort was "
            "tested and rejected as a redistributed subset of the training data. Every "
            "performance figure comes from one held-out split of a single dataset."
        )
    elif ext.get("status") == "never_attempted":
        out.append(
            "This model has no external validation. Every performance figure comes from "
            "one held-out split of a single dataset, so nothing establishes that it "
            "transfers to another cohort, site or population."
        )
    return out


# ------------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------------


def _band_info(bundle: ModelBundle, probability: float) -> dict:
    for label, lo, hi in risk_bands(bundle.threshold):
        if lo <= probability < hi:
            return {"label": label, "lower": lo, "upper": min(hi, 1.0), "note": RISK_BAND_NOTE}
    return {"label": "unknown", "lower": 0.0, "upper": 1.0, "note": RISK_BAND_NOTE}


def _explanation(bundle: ModelBundle, row: pd.DataFrame) -> dict | None:
    if bundle.explainer is None:
        return None
    local = bundle.explainer.explain_one(row)
    return {
        "method": "SHAP, aggregated from the transformed columns back to original features",
        "explainer": bundle.explainer.kind,
        "base_value": local.base_value,
        "uncalibrated_probability": local.predicted_probability,
        "note": SHAP_NOTE,
        "contributions": local.contributions,
    }


# Every field is optional and missing values are imputed, so an empty body would be scored
# entirely from training medians and returned as a prediction. A partially specified case is
# a legitimate question; a wholly unspecified one is not. Shared by /predict and /analyses so
# both refuse it identically — a save path that accepted what the predict path rejects would
# put the imputed training median profile in someone's history as their own result.
EMPTY_CASE_DETAIL = (
    "No features supplied. Every field is optional and missing values are imputed, but a "
    "request with no values at all describes no case: the result would be the imputed "
    "training median profile, not a prediction. Supply at least one feature."
)


def is_empty_case(values: dict) -> bool:
    return all(v is None for v in values.values())


def predict(bundle: ModelBundle, payload: dict, *, include_explanation: bool = True) -> dict:
    """Score one case. Never fits anything; the pipeline was fitted at training time."""
    row, imputed, extrapolated = build_row(bundle, payload)
    probability = float(bundle.pipeline.predict_proba(row)[:, 1][0])
    threshold = bundle.threshold

    warnings = _warnings(bundle, imputed, extrapolated)
    if include_explanation and bundle.explainer is None:
        warnings.append(
            "Explanation unavailable: the SHAP explainer failed to load for this model "
            f"({bundle.explainer_error}). The prediction is unaffected."
        )

    subgroup = _subgroup_performance(bundle, row)
    if subgroup:
        warnings.append(subgroup["note"])

    return {
        "disease": bundle.disease,
        "module": bundle.serving["module"],
        "positive_class_meaning": bundle.serving["positive_class_meaning"],
        "probability": probability,
        "flagged": bool(probability >= threshold),
        "threshold": threshold,
        "threshold_rule": bundle.serving["threshold_rule"],
        "risk_band": _band_info(bundle, probability),
        "model_name": bundle.serving["model"],
        "calibration": bundle.serving["calibration"],
        "model_version": bundle.version,
        "n_features_expected": len(bundle.feature_order),
        "n_features_provided": len(bundle.feature_order) - len(imputed),
        "imputed_features": imputed,
        "extrapolated_features": extrapolated,
        "warnings": warnings,
        "explanation": _explanation(bundle, row) if include_explanation else None,
        "disclaimer": DISCLAIMER,
    }


# ------------------------------------------------------------------------------------
# Scenario
# ------------------------------------------------------------------------------------


def scenario(bundle: ModelBundle, features: dict, overrides: dict, *,
             include_explanation: bool = True) -> dict:
    """Score a case, then score it again with ``overrides`` applied. Both are returned."""
    modified = {**features, **overrides}
    baseline_out = predict(bundle, features, include_explanation=include_explanation)
    modified_out = predict(bundle, modified, include_explanation=include_explanation)

    changed = []
    for name, new in overrides.items():
        old = features.get(name)
        if old != new:
            changed.append({
                "feature": name,
                "from": old,
                "to": new,
                "description": bundle.features[name]["description"],
            })

    warnings = [SCENARIO_LABEL]
    flags = NON_ACTIONABLE.get(bundle.disease, {})
    for c in changed:
        if c["feature"] in flags:
            warnings.append(
                f"'{c['feature']}' was changed, but {flags[c['feature']]}. The difference "
                "below is how the model's score moves, not an achievable change in risk."
            )
    if not changed:
        warnings.append("No feature actually changed; both estimates are the same case.")

    return {
        "disease": bundle.disease,
        "module": bundle.serving["module"],
        "label": SCENARIO_LABEL,
        "baseline": baseline_out,
        "modified": modified_out,
        "changed_features": changed,
        "probability_delta": modified_out["probability"] - baseline_out["probability"],
        "band_changed": (
            modified_out["risk_band"]["label"] != baseline_out["risk_band"]["label"]
        ),
        "flag_changed": modified_out["flagged"] != baseline_out["flagged"],
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }


# ------------------------------------------------------------------------------------
# Model summary + analytics
# ------------------------------------------------------------------------------------


def model_summary(bundle: ModelBundle) -> dict:
    s, m = bundle.serving, bundle.metadata
    return {
        "disease": bundle.disease,
        "module": s["module"],
        "positive_class_meaning": s["positive_class_meaning"],
        "model_name": s["model"],
        "calibration": s["calibration"],
        "version": bundle.version,
        "n_train": s["n_train"],
        "n_test": s["n_test"],
        "roc_auc_test": s["roc_auc_test"],
        "operating_threshold": bundle.threshold,
        "external_validation": s["external_validation"],
        "explainer": s["explainer"],
        "limitations": m.get("limitations", []),
        "disclaimer": DISCLAIMER,
    }


def _read_json(path) -> dict | list | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_csv(path) -> list[dict] | None:
    if not path.exists():
        return None
    return pd.read_csv(path).to_dict(orient="records")


def analytics(bundle: ModelBundle) -> dict:
    """Saved evaluation artifacts, served as JSON. Nothing is recomputed."""
    reports = REPORTS_DIR / bundle.disease
    shap_global = _read_json(reports / "shap_global.json") or {}
    figures = sorted(p.name for p in (reports / "figures").glob("*.png"))
    return {
        "disease": bundle.disease,
        "module": bundle.serving["module"],
        "source": (
            "Read verbatim from the artifacts written by the training run "
            f"(reports/{bundle.disease}/). No metric is recomputed at request time."
        ),
        "metrics": _read_json(reports / "metrics.json"),
        "model_comparison": _read_csv(reports / "model_comparison.csv"),
        "threshold_sweep": _read_csv(reports / "threshold_sweep.csv"),
        "shap_global_importance": shap_global.get("global_importance"),
        "shap_meta": {
            k: shap_global.get(k)
            for k in ("explainer", "base_value", "additivity_max_error",
                      "n_rows_explained", "n_test_rows", "explained_on")
        },
        "calibration": _read_json(reports / "calibration" / "summary.json"),
        "fairness": _read_json(reports / "fairness" / "disparities.json"),
        "figures": figures,
        "figure_url_template": f"/analytics/{bundle.disease}/figures/{{name}}",
        "disclaimer": DISCLAIMER,
    }

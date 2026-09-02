"""Model cards — ``models/<disease>/metadata.json`` plus a readable ``MODEL_CARD.md``.

A model card records what the model is, how it was built, how it actually performed,
and — most importantly here — what it must not be used for. Every performance number
written by :func:`build_model_card` is passed in from a real evaluation; this module
never invents or defaults a metric.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from config import (
    CV_FOLDS,
    DISCLAIMER,
    DISEASES,
    MODELS_DIR,
    RANDOM_STATE,
    RISK_BANDS,
    TEST_SIZE,
)
from ml.reporting import df_to_markdown, dict_to_markdown, write_json, write_lines

INTENDED_USE = (
    "Research and education: exploring how machine-learning models relate routinely "
    "collected clinical or survey variables to a recorded disease label, and how those "
    "relationships can be explained (SHAP). Suitable for teaching, methodology "
    "demonstrations and portfolio work."
)

OUT_OF_SCOPE_USE = [
    "Any clinical decision, diagnosis, triage, screening or treatment planning.",
    "Use on an individual patient to determine whether they have, or will develop, a disease.",
    "Deployment in a healthcare setting, or as a substitute for a qualified clinician.",
    "Populations, care settings or measurement protocols unlike the training cohort.",
]

# Hand-authored, dataset-specific caveats. Kept here (not generated) because they are
# judgements about the data, not computed values.
_LIMITATIONS: dict[str, list[str]] = {
    "heart": [
        "Very small sample: 303 patients from a single site (Cleveland Clinic), collected in the late 1980s. "
        "Confidence intervals on every metric are wide and the test set is ~61 patients.",
        "Case mix is referral-based (patients already sent for angiography), so the positive rate (~46%) "
        "is far higher than in a general population. Predicted probabilities reflect that cohort, not the public.",
        "'ca' (4 missing) and 'thal' (2 missing) are imputed inside the pipeline; both are strong predictors.",
        "'chol' contains 0 values that are almost certainly 'not measured' sentinels rather than true zeros; "
        "they are left as-is and not treated as missing.",
        "Features such as 'thal', 'ca' and 'oldpeak' come from tests (fluoroscopy, exercise ECG) that are "
        "themselves ordered because disease is already suspected — the model is not a general-population screen.",
        "Encodings for 'cp', 'slope' and 'thal' follow the Cleveland processed release; other heart datasets "
        "code these differently.",
    ],
    "kidney": [
        "Very small sample: 400 records from a single Indian hospital over a two-month period.",
        "Heavy missingness (60.5% of rows have at least one missing value) imputed inside the pipeline.",
        "The classes are close to linearly separable on a few laboratory values, so metrics are optimistically high "
        "and do not indicate the model would generalise to a screening population.",
        "Labels reflect an existing clinical diagnosis, so the model largely re-derives the diagnostic criteria "
        "rather than predicting anything ahead of time.",
    ],
    "diabetes": [
        "Features are self-reported survey answers (CDC BRFSS 2015), not laboratory measurements. "
        "The target is a self-reported diagnosis of prediabetes or diabetes, so undiagnosed cases are labelled negative.",
        "This estimates a RISK ASSOCIATION with reported diabetes status, not a diagnosis and not a future-onset prediction — "
        "the data is cross-sectional, so no temporal ordering between features and outcome is established.",
        "Survey weighting is not applied, so the sample is not nationally representative.",
        "25,772 rows share an identical feature vector with another row; the train/test split groups these so "
        "identical profiles never straddle the split.",
    ],
}


def build_model_card(
    disease: str,
    *,
    selected_model: str,
    hyperparameters: dict,
    calibration_method: str,
    calibration_rationale: str,
    selection_rationale: str,
    selection_margin: dict,
    dataset_stats: dict,
    split_meta: dict,
    cv_summary: dict,
    test_metrics: dict,
    alt_threshold_metrics: dict,
    calibration_table: pd.DataFrame,
    shap_top_features: list[dict],
    external_validation: dict | None = None,
    artifacts: dict | None = None,
) -> dict:
    """Assemble the model-card dict. Every metric argument must come from a real run."""
    info = DISEASES[disease]
    return {
        "module": info["label"],
        "disease": disease,
        "created": date.today().isoformat(),
        "disclaimer": DISCLAIMER,
        "intended_use": INTENDED_USE,
        "out_of_scope_use": OUT_OF_SCOPE_USE,
        "dataset": {
            "source": "UCI Machine Learning Repository",
            "uci_id": info["uci_id"],
            "positive_class_meaning": info["positive"],
            **dataset_stats,
        },
        "protocol": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "split_strategy": split_meta.get("strategy", split_meta.get("split_strategy")),
            "split": split_meta,
            "preprocessing": (
                "Inside the model pipeline only: numeric = median imputation + standard "
                "scaling; categorical = most-frequent imputation + one-hot "
                "(handle_unknown='ignore'); binary = most-frequent imputation. Fitted on "
                "training folds only — never on the full dataset."
            ),
            "selection_criteria": (
                "Composite of cross-validated ROC-AUC (0.4), PR-AUC (0.4) and recall (0.2); "
                "accuracy is reported but not used to select."
            ),
        },
        "model": {
            "algorithm": selected_model,
            "hyperparameters": hyperparameters,
            "calibration": calibration_method,
            "calibration_rationale": calibration_rationale,
            "selection_rationale": selection_rationale,
            "selection_margin": selection_margin,
        },
        "performance": {
            "note": (
                "All figures below are measured on the held-out test set (or by "
                "cross-validation on the training set where stated). None are estimates."
            ),
            "cross_validation_training_set": cv_summary,
            "test_set_threshold_0.5": test_metrics,
            "test_set_alternative_threshold": alt_threshold_metrics,
            "calibration_comparison": calibration_table.reset_index().to_dict(orient="records"),
            "external_validation": external_validation,
        },
        "explainability": {
            "method": "SHAP, aggregated from the transformed columns back to original features",
            "top_features": shap_top_features,
        },
        "risk_bands": {
            "note": (
                "Illustrative presentation buckets for a model-estimated probability. "
                "Not clinical decision thresholds; they carry no medical meaning."
            ),
            "bands": [{"label": n, "min": lo, "max": hi} for n, lo, hi in RISK_BANDS],
        },
        "limitations": _LIMITATIONS.get(disease, []),
        "artifacts": artifacts or {},
    }


def write_model_card(disease: str, card: dict) -> tuple[Path, Path]:
    """Write ``metadata.json`` and a human-readable ``MODEL_CARD.md``. Returns both paths."""
    out_dir = MODELS_DIR / disease
    json_path = write_json(out_dir / "metadata.json", card)
    md_path = write_lines(out_dir / "MODEL_CARD.md", _card_markdown(card))
    return json_path, md_path


def _card_markdown(card: dict) -> list[str]:
    perf = card["performance"]
    lines = [
        f"# Model card — {card['module']}",
        "",
        f"> **{card['disclaimer']}**",
        "",
        f"_Created {card['created']}._",
        "",
        "## Intended use",
        "",
        card["intended_use"],
        "",
        "### Out of scope",
        "",
        *[f"- {u}" for u in card["out_of_scope_use"]],
        "",
        "## Dataset",
        "",
        dict_to_markdown(card["dataset"], key_header="field", value_header="value"),
        "",
        "## Model",
        "",
        dict_to_markdown(
            {
                k: v
                for k, v in card["model"].items()
                if k not in ("hyperparameters", "selection_margin")
            },
            key_header="field",
            value_header="value",
        ),
        "",
        *_margin_lines(card["model"].get("selection_margin") or {}),
        "Hyperparameters:",
        "",
        "```json",
        _pretty(card["model"]["hyperparameters"]),
        "```",
        "",
        "## Protocol",
        "",
        dict_to_markdown(
            {k: v for k, v in card["protocol"].items() if k != "split"},
            key_header="field",
            value_header="value",
        ),
        "",
        "## Performance",
        "",
        f"_{perf['note']}_",
        "",
        "### Cross-validation (training set)",
        "",
        dict_to_markdown(perf["cross_validation_training_set"], key_header="metric", value_header="value"),
        "",
        "### Held-out test set (threshold 0.5)",
        "",
        dict_to_markdown(perf["test_set_threshold_0.5"], key_header="metric", value_header="value"),
        "",
        "### Held-out test set (sensitivity-oriented threshold)",
        "",
        dict_to_markdown(perf["test_set_alternative_threshold"], key_header="metric", value_header="value"),
        "",
        "### Calibration comparison",
        "",
        df_to_markdown(pd.DataFrame(perf["calibration_comparison"])),
        "",
        "## Explainability",
        "",
        f"Method: {card['explainability']['method']}",
        "",
        df_to_markdown(pd.DataFrame(card["explainability"]["top_features"])),
        "",
        "## Limitations",
        "",
        *[f"- {l}" for l in card["limitations"]],
        "",
        "## Risk bands",
        "",
        f"_{card['risk_bands']['note']}_",
        "",
        df_to_markdown(pd.DataFrame(card["risk_bands"]["bands"]), floatfmt="{:.2f}"),
        "",
        "---",
        "*Generated by `ml.model_card.write_model_card`.*",
    ]
    return lines


def _margin_lines(margin: dict) -> list[str]:
    """A prominent warning when the winning model only won by noise."""
    if not margin or margin.get("runner_up") is None:
        return []
    if not margin.get("gap_within_cv_noise"):
        return [
            f"Chosen over `{margin['runner_up']}` by {margin['gap']:.4f} cross-validated "
            f"ROC-AUC, which exceeds the fold-to-fold standard deviation "
            f"({margin['cv_roc_auc_std']:.4f}).",
            "",
        ]
    return [
        f"> **The margin of selection is not meaningful.** `{margin['selected']}` beat "
        f"`{margin['runner_up']}` by {margin['gap']:.4f} cross-validated ROC-AUC, while the "
        f"fold-to-fold standard deviation is {margin['cv_roc_auc_std']:.4f} — far larger. "
        "Treat the candidate models as performing comparably; a different random seed "
        "could easily reorder them. This choice should not be read as evidence that this "
        "algorithm is better suited to the problem.",
        "",
    ]


def _pretty(obj) -> str:
    import json

    from ml.reporting import _json_default

    return json.dumps(obj, indent=2, default=_json_default, sort_keys=True)

"""Phase 6 — external validation of the heart model against Statlog (UCI id 145).

    python -m ml.external.heart_statlog

The intended procedure was: align Statlog's 13 features to the Cleveland schema, load
``models/heart/pipeline.joblib``, predict **without retraining**, and report how the model
holds up on a second cohort.

Two things were checked before believing any number, in this order:

1. **Encoding compatibility** — the assumption carried since Phase 1 that ``cp``, ``slope``
   and ``thal`` use the same codes in both releases. This **holds**: the value domains are
   identical (cp 1-4, restecg 0-2, slope 1-3, ca 0-3, thal 3/6/7) and the marginal
   distributions agree to within about one percentage point.

2. **Dataset independence** — whether the model has already seen these rows. This **fails**.
   All 270 Statlog rows match a Cleveland row exactly on all 13 features, all 270 labels
   agree, and 222 of them (82.2%) are in the heart model's own training split. Statlog is a
   270-row subset of the Cleveland database, redistributed under a different name; it is not
   a second cohort.

So there is **no valid external validation for the heart model**, and this module reports
that rather than publishing a number. The metrics are still computed and written, clearly
labelled as contaminated, because the interesting part is how convincing they look: 0.9399
ROC-AUC, slightly *below* the internal test score, which is exactly the modest confirmation
a genuine external validation would produce. A fake that looks broken is harmless. This one
does not look broken.

The check lives in :mod:`ml.external.dataset_overlap` and is reusable — it should be run
against any future candidate before that dataset is described as external.
"""
from __future__ import annotations

import joblib
import pandas as pd

from config import MODELS_DIR, REPORTS_DIR
from ml.data.loaders import load
from ml.evaluation.metrics import classification_metrics
from ml.external.dataset_overlap import dataset_overlap, schema_compatibility
from ml.reporting import df_to_markdown, dict_to_markdown, write_json, write_lines
from ml.training.splits import make_split

DISEASE = "heart"
EXTERNAL = "statlog"


def run_external_validation(*, write: bool = True) -> dict:
    """Attempt external validation and return a structured result.

    The returned dict always carries a ``status``: ``"validated"`` if the candidate proved
    independent, ``"rejected"`` if it did not. Callers must check it — the metrics block is
    populated either way, so using it without reading ``status`` would publish a
    contaminated result.
    """
    X_int, y_int, spec = load(DISEASE)
    X_ext, y_ext, ext_spec = load(EXTERNAL)

    split = make_split(DISEASE, X_int, y_int)
    X_train, X_test, y_train, y_test = split.apply(X_int, y_int)

    # 1. Do the two releases encode the same variables the same way?
    schema = schema_compatibility(X_int, X_ext, categorical=list(spec.categorical) + list(spec.binary))
    schema_ok = bool(schema["compatible"].all())

    # 2. Has the model already seen these rows?
    overlap = dataset_overlap(
        X_ext, y_ext, X_int, y_int, X_train=X_train, X_test=X_test, columns=list(X_int.columns)
    )

    # 3. Score anyway — but only ever reported alongside the verdict above.
    pipeline = joblib.load(MODELS_DIR / DISEASE / "pipeline.joblib")
    y_prob = pipeline.predict_proba(X_ext)[:, 1]
    metrics = classification_metrics(y_ext, y_prob, threshold=0.5)

    status = "validated" if (schema_ok and overlap.independent) else "rejected"
    result = {
        "disease": DISEASE,
        "external_dataset": EXTERNAL,
        "status": status,
        "usable_as_external_validation": status == "validated",
        "schema_compatible": schema_ok,
        "schema_check": schema.to_dict(orient="records"),
        "independence_check": overlap.to_dict(),
        "verdict": overlap.verdict,
        "metrics": metrics,
        "metrics_interpretation": _interpretation(status, overlap, metrics),
        "internal_test_comparison": {
            "internal_test_n": int(len(y_test)),
            "external_n": int(len(y_ext)),
            "note": (
                "Compare only if status == 'validated'. Otherwise the external column is "
                "measuring the training set."
            ),
        },
    }

    if write:
        out = REPORTS_DIR / DISEASE
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "external_validation.json", result)
        schema.to_csv(out / "external_schema_check.csv", index=False)
        write_lines(out / "EXTERNAL_VALIDATION.md", _markdown(result, schema, metrics))
        _amend_model_card(result)
    return result


def _amend_model_card(result: dict) -> None:
    """Write the verdict into the existing model card without retraining.

    External validation runs after training, so the card on disk predates this result.
    Rebuilding it from ``metadata.json`` keeps ``MODEL_CARD.md`` and ``metadata.json``
    consistent with each other and avoids a 'validation was never attempted' reading.
    ``ml.training.finalize`` reloads the same file on a future retrain, so the finding
    survives either order.
    """
    import json

    from ml.model_card import write_model_card

    meta_path = MODELS_DIR / DISEASE / "metadata.json"
    if not meta_path.exists():
        return
    card = json.loads(meta_path.read_text(encoding="utf-8"))
    card.setdefault("performance", {})["external_validation"] = result
    write_model_card(DISEASE, card)


def _interpretation(status: str, overlap, metrics: dict) -> str:
    if status == "validated":
        return (
            f"Statlog is independent of the training data ({overlap.pct_unseen}% of rows "
            f"unseen), so ROC-AUC {metrics['roc_auc']:.4f} on n={metrics['n']} is a genuine "
            "estimate of how the model transfers to this cohort."
        )
    return (
        f"THESE NUMBERS ARE NOT EXTERNAL VALIDATION. ROC-AUC {metrics['roc_auc']:.4f} on "
        f"n={metrics['n']} was computed on rows the model was largely trained on "
        f"({overlap.n_in_train}/{overlap.n_candidate} are training rows, "
        f"{overlap.n_unseen} are unseen). They are recorded only to show how plausible a "
        "contaminated result looks: it sits just below the internal test score, which is "
        "what a sound external validation would also produce. Do not cite them."
    )


def _markdown(result: dict, schema: pd.DataFrame, metrics: dict) -> list[str]:
    ov = result["independence_check"]
    rejected = result["status"] == "rejected"
    lines = [
        "# External validation — heart model vs Statlog (UCI id 145)",
        "",
        f"**Status: {result['status'].upper()}** — "
        + ("this dataset cannot be used to validate the heart model."
           if rejected else "this dataset is usable as external validation."),
        "",
        "## Verdict",
        "",
        f"> {result['verdict']}",
        "",
        "## 1. Encoding compatibility (the Phase 1 assumption)",
        "",
        "Carried since Phase 1: that `cp`, `slope` and `thal` use the same codes in the "
        "Statlog release as in the Cleveland processed release. Aligning columns by name "
        "without checking this would align silently and predict nonsense.",
        "",
        f"**Result: {'compatible' if result['schema_compatible'] else 'INCOMPATIBLE'}.**",
        "",
        df_to_markdown(schema),
        "",
        "## 2. Dataset independence",
        "",
        "External validation only means anything if the model has not seen the rows. "
        "Matching is exact on all 13 features after numeric normalisation.",
        "",
        dict_to_markdown(
            {k: v for k, v in ov.items() if k not in ("reasons", "verdict")},
            key_header="check", value_header="value",
        ),
        "",
    ]
    if ov.get("reasons"):
        lines += ["Reasons this dataset is not independent:", ""]
        lines += [f"- {r.capitalize()}." for r in ov["reasons"]]
        lines.append("")
    lines += [
        "## 3. Metrics — recorded, not to be cited",
        "",
        f"> {result['metrics_interpretation']}",
        "",
        dict_to_markdown(metrics, key_header="metric", value_header="value"),
        "",
    ]
    if rejected:
        lines += [
            "## What this means for the heart model",
            "",
            "The heart model has **no external validation**. Its only honest performance "
            "estimate remains the 61-row held-out Cleveland test set, with all the "
            "limitations already recorded in the model card: a single-site, referral-based "
            "cohort from the late 1980s, far too small for narrow confidence intervals.",
            "",
            "This is a negative result and it is the correct one. Publishing the Statlog "
            "numbers would have manufactured false evidence of generalisation, and because "
            "they land just below the internal test score they would not have looked wrong.",
            "",
            "Genuine external validation would need a heart cohort that is not derived from "
            "the Cleveland database — for example the Hungarian, Switzerland or Long Beach "
            "partitions distributed alongside it, each of which would first have to pass "
            "this same independence check.",
            "",
        ]
    lines += [
        "---",
        "*Generated by `ml.external.heart_statlog.run_external_validation`. Every number "
        "above comes from that run.*",
    ]
    return lines


def main() -> int:
    result = run_external_validation(write=True)
    ov = result["independence_check"]
    print(f"status                : {result['status'].upper()}")
    print(f"schema compatible     : {result['schema_compatible']}")
    print(f"rows matched          : {ov['n_matched']}/{ov['n_candidate']} ({ov['pct_matched']}%)")
    print(f"  of which in training: {ov['n_in_train']}")
    print(f"  unseen by the model : {ov['n_unseen']}")
    print()
    print(result["verdict"])
    print()
    print(result["metrics_interpretation"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

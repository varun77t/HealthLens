"""Phase 7 — subgroup performance and calibration reporting for every trained model.

    python -m scripts.fairness_report

Reads the shipped pipelines and describes their behaviour on the held-out test set. No
model is trained, refitted or changed; nothing here can alter a metric reported elsewhere.

Writes per disease:
    reports/<disease>/fairness/subgroups.csv        one row per subgroup per attribute
    reports/<disease>/fairness/disparities.json     the comparison and its verdict
    reports/<disease>/fairness/FAIRNESS.md          the write-up
    reports/<disease>/calibration/summary.json      before/after Brier + ECE
    reports/<disease>/figures/calibration_before_after.png

The headline result is mostly negative and that is the point. Heart has 61 test patients
and kidney 80, so almost every subgroup falls under the minimum class counts and the
correct output is "not estimable" rather than a number that looks like a finding. Diabetes,
with 50,961 test rows, is the only module where subgroup estimates are stable — and it is
therefore the only one where a disparity can be asserted or ruled out.

**This never labels a model fair.** Subgroup parity on a test set is not fairness: the
attributes available here are sex and age only, the labels carry their own measurement
bias, and equal metrics across two groups say nothing about who the model harms.
"""
from __future__ import annotations

import sys

import joblib
import pandas as pd

from config import DISEASES, MODELS_DIR, REPORTS_DIR
from ml.data.loaders import load
from ml.evaluation.calibration import calibration_summary, plot_reliability_comparison
from ml.fairness.subgroup_metrics import (
    MIN_CLASS_N,
    MIN_SUBGROUP_N,
    PRACTICAL_GAP,
    available_attributes,
    disparity_assessment,
    subgroup_table,
)
from ml.reporting import df_to_markdown, write_json, write_lines
from ml.training.splits import make_split

METRICS_TO_COMPARE = ("recall", "specificity", "precision", "roc_auc")


def _load_predictions(disease: str):
    """Test-set predictions from the shipped pipeline and its uncalibrated base."""
    X, y, spec = load(disease)
    split = make_split(disease, X, y)
    X_train, X_test, y_train, y_test = split.apply(X, y)

    calibrated = joblib.load(MODELS_DIR / disease / "pipeline.joblib")
    base = joblib.load(MODELS_DIR / disease / "base_pipeline.joblib")
    p_cal = calibrated.predict_proba(X_test)[:, 1]
    p_raw = base.predict_proba(X_test)[:, 1]
    return X_test, y_test, p_raw, p_cal


def run_disease(disease: str, *, write: bool = True) -> dict:
    import json

    meta = json.loads((MODELS_DIR / disease / "metadata.json").read_text(encoding="utf-8"))
    method = meta["model"]["calibration"]

    metrics_json = json.loads((REPORTS_DIR / disease / "metrics.json").read_text(encoding="utf-8"))
    alt_threshold = float(metrics_json["test_set_alternative_threshold"]["threshold"])

    X_test, y_test, p_raw, p_cal = _load_predictions(disease)
    reports = REPORTS_DIR / disease
    fairness_dir = reports / "fairness"

    # --- calibration reporting -------------------------------------------------------
    cal = calibration_summary(y_test, p_raw, p_cal, method=method)

    # --- subgroup analysis, at both operating points ---------------------------------
    # Threshold-dependent metrics are reported at 0.5 *and* at the model's sensitivity
    # oriented threshold. On diabetes the two differ enormously (0.5 vs 0.1388), and a
    # subgroup recall gap at 0.5 there is mostly an artefact of where the cut-off falls
    # relative to each age band's scores. Showing both makes that visible instead of
    # letting it masquerade as a disparity.
    attributes = available_attributes(X_test, disease)
    thresholds = {"0.5": 0.5, "operating": alt_threshold}
    tables, disparities = {}, {}
    for attr, groups in attributes.items():
        per_threshold = {}
        for tname, tval in thresholds.items():
            table = subgroup_table(y_test, p_cal, groups, threshold=tval)
            table.insert(0, "attribute", attr)
            table.insert(1, "threshold", round(tval, 4))
            table.insert(2, "threshold_name", tname)
            per_threshold[tname] = table
        tables[attr] = per_threshold
        disparities[attr] = {
            tname: {m: disparity_assessment(tbl, m) for m in METRICS_TO_COMPARE}
            for tname, tbl in per_threshold.items()
        }

    combined = (
        pd.concat([t for pt in tables.values() for t in pt.values()], ignore_index=True)
        if tables else pd.DataFrame()
    )
    missing = _missing_attributes(disease, attributes)

    at_half = combined[combined["threshold_name"] == "0.5"] if len(combined) else combined
    result = {
        "disease": disease,
        "test_n": int(len(y_test)),
        "attributes_analysed": sorted(attributes),
        "attributes_unavailable": missing,
        "reliability_thresholds": {"min_subgroup_n": MIN_SUBGROUP_N, "min_class_n": MIN_CLASS_N},
        "decision_thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "n_subgroups": int(len(at_half)),
        "n_reliable_subgroups": int(at_half["reliable"].sum()) if len(at_half) else 0,
        "disparities": disparities,
        "calibration": cal,
    }

    if write:
        fairness_dir.mkdir(parents=True, exist_ok=True)
        (reports / "calibration").mkdir(parents=True, exist_ok=True)
        if len(combined):
            combined.to_csv(fairness_dir / "subgroups.csv", index=False)
        write_json(fairness_dir / "disparities.json", result)
        write_json(reports / "calibration" / "summary.json", cal)
        plot_reliability_comparison(
            y_test,
            {"raw (uncalibrated)": p_raw, f"calibrated ({method})": p_cal},
            reports / "figures",
            title=f"— {DISEASES[disease]['label']}",
        )
        write_lines(fairness_dir / "FAIRNESS.md", _markdown(disease, result, tables, cal))
    return result


def _missing_attributes(disease: str, attributes: dict) -> list[dict]:
    out = []
    if "sex" not in attributes:
        out.append({
            "attribute": "sex",
            "reason": "not recorded in this dataset — no sex breakdown is possible, and "
                      "its absence cannot be worked around without inventing data",
        })
    if "age_band" not in attributes:
        out.append({"attribute": "age_band", "reason": "no age column in this dataset"})
    return out


def _markdown(disease: str, result: dict, tables: dict, cal: dict) -> list[str]:
    label = DISEASES[disease]["label"]
    lines = [
        f"# Subgroup performance — {label}",
        "",
        "> **For research and educational purposes only.** This is a description of how one "
        "model behaved on one small held-out sample. It is not an audit, and it does not "
        "establish that the model is fair or safe to use.",
        "",
        f"Held-out test set: **n = {result['test_n']}**. Every figure below comes from the "
        "shipped pipeline's predictions on those rows; no model was retrained.",
        "",
        "## How to read this",
        "",
        f"A subgroup is marked **reliable** only if it holds at least {MIN_SUBGROUP_N} rows "
        f"with at least {MIN_CLASS_N} positives and {MIN_CLASS_N} negatives. Rates carry "
        "**Wilson 95% intervals**; ROC-AUC carries a Hanley–McNeil interval.",
        "",
        "A gap is called a **material difference** only when it clears two separate bars: "
        "the intervals must be disjoint (so it is not sampling noise), **and** the gap must "
        f"be at least {PRACTICAL_GAP}. The second bar matters as much as the first — with "
        "50,961 test rows the intervals become narrow enough that almost any difference "
        "separates, so statistical clarity stops implying importance. Small-sample noise "
        "and large-sample over-sensitivity are opposite failures and both are named here.",
        "",
        "Equal metrics across groups would **not** demonstrate fairness. Only sex and age "
        "are available here, the labels carry their own measurement bias, and parity on two "
        "attributes says nothing about who a model harms.",
        "",
    ]

    if result["attributes_unavailable"]:
        lines += ["### Attributes that could not be analysed", ""]
        lines += [f"- **{a['attribute']}** — {a['reason']}." for a in result["attributes_unavailable"]]
        lines.append("")

    lines += [
        f"**{result['n_reliable_subgroups']} of {result['n_subgroups']} subgroups meet the "
        "minimum counts.**",
        "",
    ]

    cols = ["subgroup", "n", "n_positive", "n_negative", "prevalence",
            "recall", "recall_lo", "recall_hi",
            "specificity", "specificity_lo", "specificity_hi",
            "roc_auc", "roc_auc_lo", "roc_auc_hi", "reliable"]
    for attr, per_threshold in tables.items():
        lines += [f"## By {attr}", ""]
        for tname, table in per_threshold.items():
            t = float(table["threshold"].iloc[0])
            heading = ("threshold 0.5" if tname == "0.5"
                       else f"sensitivity-oriented threshold {t:.4f}")
            lines += [f"### At {heading}", ""]
            lines += [df_to_markdown(table[[c for c in cols if c in table.columns]]), ""]
            unreliable = table[~table["reliable"]]
            if len(unreliable):
                lines += ["Caveats:", ""]
                lines += [f"- **{r['subgroup']}** — {r['caveat']}."
                          for _, r in unreliable.iterrows()]
                lines.append("")
            lines += ["Disparity assessment:", ""]
            for metric in METRICS_TO_COMPARE:
                d = result["disparities"][attr][tname][metric]
                if d.get("conclusive"):
                    mark = "**MATERIAL DIFFERENCE**"
                elif d.get("statistically_clear"):
                    mark = "small but statistically clear"
                else:
                    mark = "inconclusive"
                lines.append(f"- {mark} — {d['verdict']}")
            lines.append("")
        lines += [
            "ROC-AUC is identical across the two threshold blocks above — it does not "
            "depend on the cut-off. Only the rate metrics move, and the size of that "
            "movement is a direct measure of how much a threshold-based subgroup "
            "comparison reflects the threshold rather than the model.",
            "",
        ]
        if attr == "age_band":
            lines += [
                "One caveat specific to age: age is itself one of the model's strongest "
                "features, so computing ROC-AUC *within* an age band removes most of the "
                "variance in that feature. Within-band AUC is therefore not directly "
                "comparable to the overall test AUC. Comparing bands against each other "
                "remains meaningful, but the bands span different widths of the underlying "
                "age range, so some of the difference between them reflects that rather "
                "than model behaviour.",
                "",
            ]

    lines += [
        "## Calibration",
        "",
        f"Wrapper applied: `{cal['method_applied']}`"
        + ("" if cal["wrapper_applied"] else
           " — the base model was already the best-calibrated option on training "
           "out-of-fold Brier, so the before/after columns are identical by construction"),
        "",
        "| | Brier | ECE | ROC-AUC |",
        "|---|---|---|---|",
        f"| before (raw) | {cal['before']['brier']:.6f} | {cal['before']['ece']:.6f} | {cal['before']['roc_auc']:.4f} |",
        f"| after | {cal['after']['brier']:.6f} | {cal['after']['ece']:.6f} | {cal['after']['roc_auc']:.4f} |",
        f"| delta | {cal['delta_brier']:+.6f} | {cal['delta_ece']:+.6f} | {cal['delta_roc_auc']:+.4f} |",
        "",
        "ECE is the population-weighted mean gap between predicted probability and observed "
        "frequency across 10 bins. It isolates calibration, where Brier mixes calibration "
        "with discrimination — but ECE alone is blind to ranking, so a model predicting the "
        "base rate for everyone would score near-perfectly. Read the two together.",
        "",
        f"_{cal['note']}_",
        "",
        "See `figures/calibration_before_after.png` for the reliability curves.",
        "",
        "---",
        "*Generated by `scripts/fairness_report.py`. Every number above comes from that run.*",
    ]
    return lines


def main() -> int:
    for disease in DISEASES:
        if not (MODELS_DIR / disease / "metadata.json").exists():
            print(f"[{disease}] skipped — no trained model")
            continue
        r = run_disease(disease)
        print(f"[{disease}] test n={r['test_n']}  "
              f"subgroups {r['n_reliable_subgroups']}/{r['n_subgroups']} reliable  "
              f"attributes={r['attributes_analysed']}")
        for attr, per_threshold in r["disparities"].items():
            for tname, per_metric in per_threshold.items():
                for metric, d in per_metric.items():
                    if d.get("conclusive"):
                        print(f"    MATERIAL  {attr}/{metric} @{tname}: "
                              f"{d['worst_subgroup']} {d['worst_value']:.4f} vs "
                              f"{d['best_subgroup']} {d['best_value']:.4f} "
                              f"(gap {d['gap']:.4f})")
                    elif d.get("statistically_clear"):
                        print(f"    small     {attr}/{metric} @{tname}: gap "
                              f"{d['gap']:.4f} (statistically clear, below the "
                              f"{PRACTICAL_GAP} practical threshold)")
        cal = r["calibration"]
        print(f"    calibration {cal['method_applied']}: Brier "
              f"{cal['before']['brier']:.4f} -> {cal['after']['brier']:.4f}, "
              f"ECE {cal['before']['ece']:.4f} -> {cal['after']['ece']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

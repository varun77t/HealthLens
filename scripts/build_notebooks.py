"""Generate the thin EDA and training notebooks from templates.

Each notebook is deliberately thin: it imports the reusable ``ml`` package and renders
tables and figures. All heavy logic lives in ``ml/``.

The **training** notebooks deliberately do *not* retrain. Training happens in
``scripts/train_<disease>.py``; the notebook reads the artifacts that run produced and
narrates them. That keeps the notebook fast, keeps a single source of truth for every
number, and makes it impossible for a notebook to display a metric that differs from
the one in ``reports/<disease>/metrics.json``. A training notebook is only generated
for a disease that has already been trained.

Usage:
    python -m scripts.build_notebooks
"""
from __future__ import annotations

import nbformat as nbf

from config import MODELS_DIR, NOTEBOOKS_DIR, REPORTS_DIR

EDA_CELLS = [
    ("markdown", """# {label} — Exploratory Data Analysis

> **For research and educational purposes only.** This notebook explores a historical
> public dataset. Nothing here is a medical diagnosis or medical advice.

This notebook is a thin view over the reusable `ml` package. Re-run
`python -m ml.eda.run {disease}` to regenerate `reports/{disease}/`.
"""),
    ("code", """import sys, os
# Make the repository root importable when the kernel starts in notebooks/.
_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)  # so relative paths (reports/, data/) resolve from the repo root

import pandas as pd
pd.set_option("display.max_columns", 60)

from ml.data.loaders import load_{disease}
from ml.eda import profile

X, y, spec = load_{disease}()
print(spec.notes)
"""),
    ("markdown", "## 1. Dataset overview"),
    ("code", "profile.dataset_overview(X, y, spec)"),
    ("markdown", "## 2. Target distribution"),
    ("code", """display(profile.target_distribution(y))
print("class-imbalance ratio (majority / minority):", profile.class_imbalance_ratio(y))
"""),
    ("markdown", "## 3. Missing values"),
    ("code", "profile.missing_value_table(X)"),
    ("markdown", "## 4. Numeric feature summary"),
    ("code", "profile.numeric_summary(X, spec)"),
    ("markdown", "## 5. Outlier scan (IQR fences — diagnostic only, nothing removed)"),
    ("code", "profile.outlier_scan(X, spec)"),
    ("markdown", "## 6. Numeric feature / target association (descriptive)"),
    ("code", "profile.target_feature_association(X, y, spec)"),
    ("markdown", "## 7. Categorical / binary breakdown"),
    ("code", """cs = profile.categorical_summary(X, spec)
for name, table in cs.items():
    print(f"--- {name} ---")
    display(table)
"""),
    ("markdown", "## 8. Figures\n\nGenerated into `reports/{disease}/figures/`."),
    ("code", """from pathlib import Path
from IPython.display import Image, display
from ml.eda.run import run_eda

run_eda("{disease}")
fig_dir = Path("reports") / "{disease}" / "figures"
for png in sorted(fig_dir.glob("*.png")):
    print(png.name)
    display(Image(filename=str(png)))
"""),
    ("markdown", """## 9. Written summaries

- Target definition + feature dictionary: [`reports/{disease}/TARGET.md`](../reports/{disease}/TARGET.md)
- EDA summary: [`reports/{disease}/EDA.md`](../reports/{disease}/EDA.md)
"""),
]

TRAINING_CELLS = [
    ("markdown", """# {label} — Model Training Run

> **For research and educational purposes only.** The model below is trained on a
> historical public dataset. Its output is not a medical diagnosis and not medical
> advice. See the model card for what this model must **not** be used for.

This notebook **does not train anything** — it reads the artifacts produced by

```
python -m scripts.train_{disease}
```

so every number shown here is the same number in `reports/{disease}/` and
`models/{disease}/metadata.json`. Re-run that script to refresh them.
"""),
    ("code", """import sys, os
# Make the repository root importable when the kernel starts in notebooks/.
_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

import json
from pathlib import Path

import joblib
import pandas as pd
from IPython.display import Image, Markdown, display

pd.set_option("display.max_columns", 60)

DISEASE = "{disease}"
reports = Path("reports") / DISEASE
models = Path("models") / DISEASE

metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
card = json.loads((models / "metadata.json").read_text(encoding="utf-8"))
shap_global = json.loads((reports / "shap_global.json").read_text(encoding="utf-8"))

print(card["module"])
print("algorithm :", card["model"]["algorithm"])
print("calibration:", card["model"]["calibration"])
"""),
    ("markdown", """## 1. How the data was split

Preprocessing (imputation, scaling, one-hot) happens **inside** the model pipeline, so
it is fitted on training folds only and never sees the held-out rows.
"""),
    ("code", """pd.Series(metrics["split"]).to_frame("value")"""),
    ("markdown", "## 2. Model comparison\n\nFive candidate pipelines, 5-fold cross-validated on the training set."),
    ("code", """pd.read_csv(reports / "model_comparison.csv")"""),
    ("markdown", """### Was the winner meaningfully better?

`selection_margin` compares the tuned top-2 against the fold-to-fold standard deviation.
When the gap is smaller than that noise, the leaderboard is **not** a quality ranking.
"""),
    ("code", """pd.Series(metrics["selection_margin"]).to_frame("value")"""),
    ("code", """display(Markdown((reports / "SELECTION.md").read_text(encoding="utf-8")))"""),
    ("markdown", """## 3. Calibration

The wrapper is chosen on **out-of-fold training** scores. The test-set table below is
reported for transparency only — choosing on it would make the test metrics optimistic.
"""),
    ("code", """print("selection (training out-of-fold):")
display(pd.DataFrame(metrics["calibration_selection_train_cv"]).set_index("method"))
print("\\nreported only (test set) — NOT used to choose:")
display(pd.DataFrame(metrics["calibration_test_comparison"]).set_index("method"))
print("\\nchosen:", card["model"]["calibration"])
print(card["model"]["calibration_rationale"])
"""),
    ("markdown", "## 4. Held-out test performance"),
    ("code", """rows = {
    "threshold 0.5": metrics["test_set_threshold_0.5"],
    "sensitivity-oriented": {k: v for k, v in metrics["test_set_alternative_threshold"].items()
                             if k != "threshold_rule"},
}
display(pd.DataFrame(rows).loc[
    ["n", "threshold", "roc_auc", "pr_auc", "recall_sensitivity", "specificity",
     "precision", "f1", "accuracy", "brier", "cm_tn", "cm_fp", "cm_fn", "cm_tp"]
])
print(metrics["test_set_alternative_threshold"]["threshold_rule"])
"""),
    ("markdown", """### Which threshold should you read?

0.5 is a convention, not a decision rule. On a low-prevalence problem a well-calibrated
model should rarely output a probability above 0.5 — most people really do have a
below-even chance — so thresholding there gives high specificity and low recall. That
looks like a broken model and is not one.
"""),
    ("code", """note = metrics.get("diagnostics", {}).get("operating_point_note")
display(Markdown(f"> {note}" if note else "_No operating-point note recorded._"))
"""),
    ("code", """for name in ["roc_curve", "pr_curve", "confusion_matrix", "calibration_curve"]:
    png = reports / "figures" / f"{name}.png"
    if png.exists():
        display(Image(filename=str(png)))
"""),
    ("markdown", "### Threshold sweep\n\nHow precision, recall and specificity trade off across the decision threshold."),
    ("code", """pd.read_csv(reports / "threshold_sweep.csv")"""),
    ("markdown", """## 4b. What actually drives this score?

A high ROC-AUC is not by itself evidence that the model learned clinical structure — it
can also come from how the data was collected. The probe below discards **every measured
value** and trains only on *which values are missing*. If that alone reproduces most of
the headline score, the model is largely reading which tests a clinician chose to order.

That is not train/test leakage — the split stays clean — but it is a ceiling on how far
the result transfers to a setting where these tests are ordered routinely.
"""),
    ("code", """diag = metrics.get("diagnostics", {})
print(diag.get("attribution_note", "(no diagnostics recorded)"))
print()

probe = diag.get("missingness_only_probe")
if probe:
    print("missingness-indicators-only model:")
    for k in ("n_indicator_features", "cv_roc_auc_mean", "cv_roc_auc_std",
              "test_roc_auc", "test_pr_auc", "test_accuracy"):
        if k in probe:
            print(f"  {k:<22} {probe[k]}")
    print(f"\\n  full model test ROC-AUC {metrics['test_set_threshold_0.5']['roc_auc']:.4f}")
else:
    print("No missing values in this dataset — measurement pattern cannot carry signal.")

print("\\ncomplete-case analysis (why it was not used instead):")
print(" ", diag.get("complete_case"))
"""),
    ("code", """miss_csv = reports / "missingness_vs_target.csv"
if miss_csv.exists():
    display(Markdown("**Is a column's missingness itself predictive of the target?**"))
    display(pd.read_csv(miss_csv))
else:
    print("No missing values to analyse.")
"""),
    ("markdown", """### 4c. What could any model achieve on these features?

When two rows carry an identical feature vector but different outcomes, no model that sees
only these features can get both right. That is irreducible error, and it caps accuracy
regardless of the algorithm. Reported so the gap to 1.0 is attributed honestly — and so
that it is *not* used to excuse a model sitting far below the bound.
"""),
    ("code", """dup = diag.get("duplicate_feature_vectors")
display(Markdown(f"> {diag.get('duplicate_conflict_note', '(not recorded)')}"))
if dup:
    display(pd.Series(dup).to_frame("value"))
"""),
    ("markdown", """### 4d. Did SMOTE actually help?

"Use SMOTE for imbalanced data" is repeated far more often than it is checked. Where this
comparison was run, the selected model was cross-validated on identical folds under SMOTE
and under class weighting. SMOTE resamples **inside** each fold's training portion only.

Watch PR-AUC rather than recall: both strategies move the operating point, so recall at a
fixed 0.5 threshold swings hard while the threshold-free ranking barely changes.
"""),
    ("code", """imb_csv = reports / "imbalance_comparison.csv"
if imb_csv.exists():
    display(Markdown(f"> {diag.get('imbalance_note', '')}"))
    display(pd.read_csv(imb_csv))
else:
    print("Imbalance strategies were not compared for this disease.")
"""),
    ("markdown", """## 5. Explainability (SHAP)

SHAP values are computed on the transformed matrix and summed back onto the **original**
feature names. `additivity_max_error` is the largest gap between `base value + sum(SHAP)`
and the model's actual output — near zero means the explanation really does reconstruct
this model rather than being a plausible-looking set of numbers.
"""),
    ("code", """print("explainer:", shap_global["explainer"])
print("additivity max error:", shap_global["additivity_max_error"])
display(pd.DataFrame(shap_global["global_importance"]))
"""),
    ("code", """for name in ["shap_global_importance", "shap_beeswarm"]:
    png = reports / "figures" / f"{name}.png"
    if png.exists():
        display(Image(filename=str(png)))
"""),
    ("markdown", """### A single explained case

Signed contributions for one held-out patient. Positive pushes the model toward the
positive class, negative away from it. This is an explanation of **the model's output**,
not a clinical account of why this person is or is not ill.
"""),
    ("code", """local = shap_global["example_local_explanation"]
print("base value           :", round(local["base_value"], 4))
print("predicted probability:", round(local["predicted_probability"], 4))
display(pd.DataFrame(local["contributions"]))
"""),
    ("markdown", """## 6. Model card

Intended use, out-of-scope use, and the limitations of this model.
"""),
    ("code", """display(Markdown((models / "MODEL_CARD.md").read_text(encoding="utf-8")))"""),
]

EXTERNAL_CELLS = [
    ("markdown", """# External Validation — Heart model vs Statlog

> **For research and educational purposes only.** Nothing here is a medical diagnosis or
> medical advice.

Internal test performance says how a model does on held-out rows from the *same* dataset.
External validation asks a harder question: does it hold up on a **different cohort**?

This notebook reads the artifacts written by

```
python -m ml.external.heart_statlog
```

**The result is a negative one, and that is the finding.** Statlog turned out not to be an
independent cohort at all, so the heart model has no external validation. The check that
established this is reusable and should be run against any future candidate dataset.
"""),
    ("code", """import sys, os
_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

import json
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

pd.set_option("display.max_columns", 60)

ext = json.loads(Path("reports/heart/external_validation.json").read_text(encoding="utf-8"))
print("status                        :", ext["status"])
print("schema compatible             :", ext["schema_compatible"])
print("usable as external validation :", ext["usable_as_external_validation"])
"""),
    ("markdown", """## 1. Encoding compatibility — the assumption carried since Phase 1

Aligning a second dataset by column *name* is only safe if the *encodings* agree. A `thal`
coded 3/6/7 in one release and 0/1/2 in another aligns silently and predicts nonsense.

This was an untested assumption until now. It **holds** — but verifying it matters for a
second reason: it is what makes the contamination finding below meaningful. The rows really
are the same records, not coincidentally equal under a mismatched encoding.
"""),
    ("code", """display(pd.DataFrame(ext["schema_check"]))"""),
    ("markdown", """## 2. Is the dataset actually independent?

Measured **before** any metric is computed. Matching is exact on all 13 features after
numeric normalisation — casting to float first, because the same record arriving as `int64`
in one release and `float64` in another would otherwise never match, silently reporting a
contaminated dataset as clean.
"""),
    ("code", """ov = ext["independence_check"]
display(pd.Series({k: v for k, v in ov.items() if not isinstance(v, list)}).to_frame("value"))
display(Markdown("**Reasons:**\\n" + "\\n".join(f"- {r.capitalize()}." for r in ov["reasons"])))
"""),
    ("code", """display(Markdown(f"> {ext['verdict']}"))"""),
    ("markdown", """## 3. The metrics — recorded, not to be cited

The model was scored on Statlog anyway. Read the number, then read what it is worth.

This is the part worth dwelling on: the contaminated result does **not** look broken. It
lands just *below* the internal test score, which is exactly the modest confirmation a
sound external validation would produce. A fake that looks obviously wrong is harmless;
this one would have passed review.
"""),
    ("code", """display(pd.Series(ext["metrics"]).to_frame("value"))
display(Markdown(f"> {ext['metrics_interpretation']}"))
"""),
    ("code", """internal = json.loads(Path("reports/heart/metrics.json").read_text(encoding="utf-8"))
compare = pd.DataFrame({
    "internal test (honest, n=61)": internal["test_set_threshold_0.5"],
    "statlog (CONTAMINATED, n=270)": ext["metrics"],
}).loc[["n", "roc_auc", "pr_auc", "recall_sensitivity", "specificity", "precision", "accuracy"]]
display(compare)
print("The two columns are close. That closeness is the trap, not the reassurance.")
"""),
    ("markdown", """## 4. What this means

The heart model has **no external validation**. Its only honest performance estimate stays
the 61-row held-out Cleveland test set, with every limitation already in the model card:
a single-site, referral-based cohort from the late 1980s, far too small for narrow
confidence intervals.

Genuine external validation would need a heart cohort not derived from the Cleveland
database — the Hungarian, Switzerland or Long Beach partitions distributed alongside it are
candidates, and each would have to pass this same independence check first.
"""),
    ("code", """display(Markdown(Path("reports/heart/EXTERNAL_VALIDATION.md").read_text(encoding="utf-8")))"""),
]

FAIRNESS_CELLS = [
    ("markdown", """# Calibration & Subgroup Performance

> **For research and educational purposes only.** This notebook describes how three models
> behaved on their held-out test sets. It is not an audit, and nothing here establishes
> that any model is fair or safe to use.

Reads the artifacts written by

```
python -m scripts.fairness_report
```

Two questions:

1. **Calibration** — when the model says 0.30, do 30% of such people actually have the
   outcome? Measured before and after the calibration wrapper.
2. **Subgroups** — does performance differ by sex or age band, and is any observed gap
   larger than sampling noise *and* large enough to matter?

**The honest answer differs sharply by dataset.** Diabetes has 50,961 test rows and
supports real conclusions. Heart has 61 and kidney 80 — there, the correct output is
"not estimable", and this notebook shows what that looks like rather than hiding it.
"""),
    ("code", """import sys, os
_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

DISEASES = ["heart", "kidney", "diabetes"]
reports = {d: json.loads(Path(f"reports/{d}/fairness/disparities.json").read_text(encoding="utf-8"))
           for d in DISEASES}
subgroups = {d: pd.read_csv(f"reports/{d}/fairness/subgroups.csv") for d in DISEASES}

pd.DataFrame([{
    "disease": d,
    "test_n": r["test_n"],
    "attributes": ", ".join(r["attributes_analysed"]),
    "reliable_subgroups": f"{r['n_reliable_subgroups']}/{r['n_subgroups']}",
} for d, r in reports.items()])
"""),
    ("markdown", """## 1. Calibration — before and after

Brier mixes calibration with discrimination. **ECE** (expected calibration error) isolates
the calibration part: within each bin of predicted probability, how far is the average
prediction from the observed frequency.

Read them together — ECE alone is blind to ranking, so a model predicting the base rate for
everybody would score a near-perfect ECE while being useless.
"""),
    ("code", """rows = []
for d in DISEASES:
    c = json.loads(Path(f"reports/{d}/calibration/summary.json").read_text(encoding="utf-8"))
    rows.append({
        "disease": d, "method": c["method_applied"],
        "brier_before": c["before"]["brier"], "brier_after": c["after"]["brier"],
        "ece_before": c["before"]["ece"], "ece_after": c["after"]["ece"],
        "delta_roc_auc": c["delta_roc_auc"],
    })
display(pd.DataFrame(rows).set_index("disease"))
print("kidney shows no change because 'raw' was selected - the base model was already the")
print("best-calibrated option on training out-of-fold Brier. That is a result, not a bug.")
"""),
    ("code", """for d in DISEASES:
    png = Path(f"reports/{d}/figures/calibration_before_after.png")
    if png.exists():
        display(Image(filename=str(png)))
"""),
    ("markdown", """## 2. Where subgroup analysis is possible — and where it is not

A subgroup is **reliable** only with at least 20 rows, 10 positives and 10 negatives.
Rates carry Wilson 95% intervals; ROC-AUC carries a Hanley-McNeil interval.
"""),
    ("code", """cols = ["attribute", "subgroup", "n", "n_positive", "n_negative",
        "recall", "recall_lo", "recall_hi", "roc_auc", "reliable"]
for d in DISEASES:
    t = subgroups[d]
    t = t[t.threshold_name == "0.5"]
    display(Markdown(f"**{d}** (test n={reports[d]['test_n']})"))
    display(t[cols])
"""),
    ("markdown", """### What the heart table shows

Three of heart's age bands report a recall of **1.0000** — with Wilson intervals like
[0.44, 1.00]. Those are not estimates; they are one, three or four patients. The `reliable`
column is what stops them being read as a finding.

Note the trap in both directions: a *small* gap on this table would be equally worthless as
evidence of equity. Too little data to show a disparity is not the same as evidence there
isn't one.
"""),
    ("markdown", """## 3. Disparity verdicts

A gap is a **material difference** only if it clears two bars: the intervals are disjoint
(not noise) *and* the gap is at least 0.05 (large enough to matter). The second bar exists
because with 50,961 rows almost any difference becomes statistically detectable.
"""),
    ("code", """rows = []
for d in DISEASES:
    for attr, per_threshold in reports[d]["disparities"].items():
        for tname, per_metric in per_threshold.items():
            for metric, x in per_metric.items():
                if x.get("n_subgroups", 0) < 2:
                    continue
                rows.append({
                    "disease": d, "attribute": attr, "threshold": tname, "metric": metric,
                    "gap": x.get("gap"),
                    "disjoint": x.get("intervals_disjoint"),
                    "reliable": x.get("all_subgroups_reliable"),
                    "material": x.get("conclusive"),
                })
verdicts = pd.DataFrame(rows)
display(verdicts)
print("material differences found:", int(verdicts["material"].sum()))
"""),
    ("markdown", """## 4. The threshold artifact

Rate metrics are shown at two cut-offs: 0.5 and each model's sensitivity-oriented
threshold. Compare how much the subgroup gaps move.

ROC-AUC is threshold-free and does **not** move. The gap between how far the rate metrics
shift and how far ROC-AUC shifts (zero) measures how much a threshold-based subgroup
comparison is telling you about the cut-off rather than the model.
"""),
    ("code", """t = subgroups["diabetes"]
t = t[t.attribute == "age_band"]
display(t.pivot(index="subgroup", columns="threshold_name",
                values=["recall", "specificity", "roc_auc"]).round(4))
"""),
    ("markdown", """## 5. Full write-ups"""),
    ("code", """for d in DISEASES:
    display(Markdown(Path(f"reports/{d}/fairness/FAIRNESS.md").read_text(encoding="utf-8")))
"""),
]

DISEASE_LABELS = {
    "heart": "Heart Disease Presence Prediction",
    "kidney": "Chronic Kidney Disease Presence Prediction",
    "diabetes": "Diabetes Health-Indicator Risk Prediction",
}


def _build(cells_template, disease: str, label: str, suffix: str) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "medicl", "display_name": "Python (medicl)", "language": "python"}
    cells = []
    for kind, src in cells_template:
        text = src.replace("{disease}", disease).replace("{label}", label)
        if kind == "markdown":
            cells.append(nbf.v4.new_markdown_cell(text))
        else:
            cells.append(nbf.v4.new_code_cell(text))
    nb["cells"] = cells
    path = NOTEBOOKS_DIR / f"{disease}_{suffix}.ipynb"
    nbf.write(nb, path)
    print(f"wrote {path}")


def build_eda_notebook(disease: str, label: str) -> None:
    _build(EDA_CELLS, disease, label, "eda")


def build_training_notebook(disease: str, label: str) -> None:
    _build(TRAINING_CELLS, disease, label, "training")


def _build_standalone(cells, filename: str) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "medicl", "display_name": "Python (medicl)", "language": "python"}
    nb["cells"] = [
        nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    path = NOTEBOOKS_DIR / filename
    nbf.write(nb, path)
    print(f"wrote {path}")


def build_external_validation_notebook() -> None:
    _build_standalone(EXTERNAL_CELLS, "external_validation.ipynb")


def build_fairness_notebook() -> None:
    _build_standalone(FAIRNESS_CELLS, "calibration_fairness.ipynb")


def main() -> None:
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    for disease, label in DISEASE_LABELS.items():
        build_eda_notebook(disease, label)
        if (MODELS_DIR / disease / "metadata.json").exists():
            build_training_notebook(disease, label)
        else:
            print(f"skipping {disease}_training.ipynb (no trained model yet)")

    if (REPORTS_DIR / "heart" / "external_validation.json").exists():
        build_external_validation_notebook()
    else:
        print("skipping external_validation.ipynb (not run yet)")

    if all((REPORTS_DIR / d / "fairness" / "disparities.json").exists() for d in DISEASE_LABELS):
        build_fairness_notebook()
    else:
        print("skipping calibration_fairness.ipynb (run scripts.fairness_report first)")


if __name__ == "__main__":
    main()

"""Train and ship the Diabetes Health-Indicator Risk Prediction model.

    python -m scripts.train_diabetes

Writes ``models/diabetes/`` (pipeline, base pipeline, metadata.json, MODEL_CARD.md) and
``reports/diabetes/`` (model_comparison.csv, SELECTION.md, metrics.json,
threshold_sweep.csv, imbalance_comparison.csv, calibration/, shap_global.json, figures/).
Deterministic given ``RANDOM_STATE = 42``.

This run differs from heart and kidney in three ways, all forced by scale and imbalance:

* **SVC is not a candidate.** Measured fit times on subsamples give an empirical exponent
  of ~2.28 in n, extrapolating to ~7.7 h for one fit on the 202,719 training rows and
  ~23 h for a single cross-validation pass. It is excluded on cost and the reason is
  recorded in ``SELECTION.md`` and the model card, so the comparison is not quietly
  narrower than it looks.
* **PR-AUC carries the weight.** At a ~14% positive rate, ROC-AUC flatters a model that
  is poor at the minority class. PR-AUC and recall are the metrics to read here.
* **SMOTE is measured, not assumed.** ``compare_imbalance=True`` cross-validates the
  selected model under both SMOTE and class weighting on identical folds and writes the
  comparison, rather than taking the usual advice on faith.

Research/education only — self-reported survey indicators, not a clinical diagnostic tool.
"""
from __future__ import annotations

import sys

from ml.training.finalize import train_disease

DISEASE = "diabetes"


def main() -> int:
    result = train_disease(
        DISEASE,
        compare_imbalance=True,   # does SMOTE actually beat class weights here?
        max_explain_rows=5000,    # SHAP over a fixed random sample of the 50,961 test rows
    )
    m = result["test_metrics"]
    alt = result["alt_threshold_metrics"]
    print()
    print(f"=== {DISEASE}: held-out test set (n={m['n']}, threshold 0.5) ===")
    for key in ("roc_auc", "pr_auc", "recall_sensitivity", "specificity",
                "precision", "f1", "accuracy", "brier"):
        print(f"  {key:<20} {m[key]:.4f}")
    print(f"  model                {result['selected_model']}")
    print(f"  calibration          {result['calibration']}")
    print(f"  SHAP additivity err  {result['shap_additivity_max_error']}")
    print()
    print(f"--- sensitivity-oriented threshold {alt['threshold']:.4f} ---")
    for key in ("recall_sensitivity", "specificity", "precision", "f1", "accuracy"):
        print(f"  {key:<20} {alt[key]:.4f}")
    print()
    print("Prevalence in the test set:", m["prevalence"])
    print("Read PR-AUC before ROC-AUC: at this positive rate a model can look strong on "
          "ROC-AUC while being weak on the minority class.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

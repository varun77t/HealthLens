"""Train and ship the Heart Disease Presence Prediction model.

    python -m scripts.train_heart

Writes ``models/heart/`` (pipeline, base pipeline, metadata.json, MODEL_CARD.md) and
``reports/heart/`` (model_comparison.csv, SELECTION.md, metrics.json, threshold_sweep.csv,
calibration/, shap_global.json, figures/). Deterministic given ``RANDOM_STATE = 42``.

Research/education only — not a clinical diagnostic tool.
"""
from __future__ import annotations

import sys

from ml.training.finalize import train_disease

DISEASE = "heart"


def main() -> int:
    result = train_disease(DISEASE)
    m = result["test_metrics"]
    print()
    print(f"=== {DISEASE}: held-out test set (n={m['n']}, threshold 0.5) ===")
    for key in ("roc_auc", "pr_auc", "recall_sensitivity", "specificity",
                "precision", "f1", "accuracy", "brier"):
        print(f"  {key:<20} {m[key]:.4f}")
    print(f"  model                {result['selected_model']}")
    print(f"  calibration          {result['calibration']}")
    print(f"  SHAP additivity err  {result['shap_additivity_max_error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

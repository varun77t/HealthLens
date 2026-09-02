"""Generate ``reports/<disease>/TARGET.md`` — the target definition + feature dictionary.

The prose caveats live here (per dataset); the feature table and target counts are
derived from the loader + a fresh profile so the document can never drift from the code.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DISCLAIMER, DISEASES, REPORTS_DIR
from ml.data.feature_spec import FeatureSpec
from ml.data.loaders import LOADERS
from ml.eda.profile import target_distribution

# Dataset-specific target definition + caveats (hand-authored, reviewed).
TARGET_DOCS: dict[str, dict] = {
    "heart": {
        "dataset": "UCI Heart Disease (Cleveland processed) — dataset id 45",
        "raw_target": "num (angiographic disease status, integer 0-4)",
        "transformation": "y = 1 if num > 0 else 0  (0 = <50% diameter narrowing, 1 = >50% narrowing in any grade 1-4)",
        "module_name": DISEASES["heart"]["label"],
        "caveats": [
            "Single-site cohort (Cleveland Clinic Foundation), n=303 — small by modern standards.",
            "'ca' (number of major vessels) and 'thal' contain missing values (originally '?'); "
            "imputation is done inside the model pipeline, never on the full dataset.",
            "The 13-feature 'processed' subset is used, matching the canonical published experiments.",
            "'ca' and 'slope' are ordinal but are modelled as categorical.",
            "This module predicts PRESENCE of angiographic disease in this historical cohort, "
            "not future onset and not a diagnosis.",
        ],
    },
    "kidney": {
        "dataset": "UCI Chronic Kidney Disease — dataset id 336",
        "raw_target": "class (text: 'ckd' or 'notckd')",
        "transformation": "y = 1 if class == 'ckd' else 0",
        "module_name": DISEASES["kidney"]["label"],
        "caveats": [
            "n=400, collected over ~2 months at a single hospital in India — small and geographically narrow.",
            "Heavy missingness: most columns have missing values (e.g. rbc ~38%, rbcc ~33%, sod/pot ~22%). "
            "Imputation happens inside the model pipeline.",
            "Numeric columns pcv/wbcc/rbcc were stored as text with stray tab characters and are coerced to numbers.",
            "Nominal columns (normal/abnormal, present/notpresent, yes/no, good/poor) are mapped to 0/1.",
            "'sg', 'al', 'su' are ordinal clinical scales kept as numeric.",
            "The classes are strongly separable in this dataset; expect optimistic metrics and interpret with care.",
        ],
    },
    "diabetes": {
        "dataset": "UCI CDC Diabetes Health Indicators (BRFSS 2015) — dataset id 891",
        "raw_target": "Diabetes_binary (0 / 1)",
        "transformation": "y = Diabetes_binary  (1 = respondent reported prediabetes OR diabetes, 0 = neither)",
        "module_name": DISEASES["diabetes"]["label"],
        "caveats": [
            "Derived from the CDC Behavioral Risk Factor Surveillance System telephone survey — "
            "features are SELF-REPORTED health indicators, not laboratory measurements.",
            "The positive class merges prediabetes and diabetes; the dataset cannot separate the two.",
            "Full imbalanced release used (~13.9% positive, n=253,680) to preserve real prevalence "
            "for calibration; the balanced 50/50 subset is deliberately NOT used.",
            "Ordinal survey codes (GenHlth 1-5, Age 1-13 bands, Education 1-6, Income 1-8) are modelled as numeric.",
            "This module estimates a RISK ASSOCIATION with reported diabetes status in survey data. "
            "It is not a screening test and not a diagnosis.",
        ],
    },
}


def _feature_table(X: pd.DataFrame, spec: FeatureSpec) -> str:
    lines = ["| Feature | Role | Missing | Unique | Description |", "|---|---|---|---|---|"]
    n = len(X)
    for col in spec.features:
        miss = int(X[col].isna().sum())
        miss_str = f"{miss} ({100 * miss / n:.1f}%)" if miss else "0"
        lines.append(
            f"| `{col}` | {spec.kind(col)} | {miss_str} | {int(X[col].nunique(dropna=True))} | {spec.describe(col)} |"
        )
    return "\n".join(lines)


def build_target_md(disease: str) -> Path:
    if disease not in TARGET_DOCS:
        raise KeyError(disease)
    X, y, spec = LOADERS[disease]()
    doc = TARGET_DOCS[disease]
    dist = target_distribution(y)
    dist_rows = "\n".join(
        f"| {int(r['class'])} | {int(r['count'])} | {r['proportion']:.4f} |" for _, r in dist.iterrows()
    )

    md = f"""# Target definition — {doc['module_name']}

> {DISCLAIMER}

## Dataset
{doc['dataset']}

- Rows: **{len(X)}**
- Predictor features: **{X.shape[1]}**  ({len(spec.numeric)} numeric, {len(spec.categorical)} categorical, {len(spec.binary)} binary)

## Prediction target

- **Raw target column:** {doc['raw_target']}
- **Transformation applied:** `{doc['transformation']}`
- **Positive class (y = 1) means:** {DISEASES[disease]['positive']}

### Target distribution (after transformation)

| class | count | proportion |
|---|---|---|
{dist_rows}

## Feature dictionary

{_feature_table(X, spec)}

## Caveats and limitations

""" + "\n".join(f"- {c}" for c in doc["caveats"]) + f"""

## Loader notes (from code)

{spec.notes}

---
*Generated by `python -m ml.data.target_docs`. Values are computed from the cached raw
dataset and the loader in `ml/data/loaders.py` — edit the code, not this file.*
"""
    out = REPORTS_DIR / disease / "TARGET.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return out


def main() -> None:
    for disease in TARGET_DOCS:
        path = build_target_md(disease)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

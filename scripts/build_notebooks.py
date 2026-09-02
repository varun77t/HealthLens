"""Generate the thin EDA notebooks from a template.

Each notebook is deliberately thin: it imports the reusable ``ml`` package, loads one
dataset, and renders the profiling tables + figures. All heavy logic lives in ``ml/``.

Usage:
    python -m scripts.build_notebooks
"""
from __future__ import annotations

import nbformat as nbf

from config import NOTEBOOKS_DIR

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

DISEASE_LABELS = {
    "heart": "Heart Disease Presence Prediction",
    "kidney": "Chronic Kidney Disease Presence Prediction",
    "diabetes": "Diabetes Health-Indicator Risk Prediction",
}


def build_eda_notebook(disease: str, label: str) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "medicl", "display_name": "Python (medicl)", "language": "python"}
    cells = []
    for kind, src in EDA_CELLS:
        text = src.replace("{disease}", disease).replace("{label}", label)
        if kind == "markdown":
            cells.append(nbf.v4.new_markdown_cell(text))
        else:
            cells.append(nbf.v4.new_code_cell(text))
    nb["cells"] = cells
    path = NOTEBOOKS_DIR / f"{disease}_eda.ipynb"
    nbf.write(nb, path)
    print(f"wrote {path}")


def main() -> None:
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    for disease, label in DISEASE_LABELS.items():
        build_eda_notebook(disease, label)


if __name__ == "__main__":
    main()

"""Central configuration for the Multi-Disease AI platform.

Everything that must stay consistent across the ML pipeline, notebooks and backend
lives here: the global random seed, filesystem paths, the disease registry, and the
(illustrative, non-clinical) risk-band thresholds.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
RANDOM_STATE: int = 42

# Fraction of each dataset held out as a final, untouched test set.
TEST_SIZE: float = 0.20

# Cross-validation protocol used for model comparison and tuning.
CV_FOLDS: int = 5

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"

# --------------------------------------------------------------------------------------
# Disease registry
# --------------------------------------------------------------------------------------
# `uci_id`     : dataset id passed to ucimlrepo.fetch_ucirepo
# `label`      : human-readable module name (never uses the word "diagnosis")
# `positive`   : meaning of the positive class (y == 1)
# `raw_subdir` : folder under data/raw/ holding the cached CSVs
DISEASES: dict[str, dict] = {
    "heart": {
        "uci_id": 45,
        "label": "Heart Disease Presence Prediction",
        "positive": "angiographic heart disease present (>50% diameter narrowing)",
        "raw_subdir": "heart",
        # Phase 6 measured Statlog (id 145) to be a 270-row subset of this very dataset
        # (100% exact feature+label match, 82.2% inside the training split), so it cannot
        # validate this model. No usable external cohort is currently available.
        "external_validation": None,
        "external_validation_rejected": {
            "statlog": "subset of the Cleveland training data — see "
                       "reports/heart/EXTERNAL_VALIDATION.md",
        },
    },
    "kidney": {
        "uci_id": 336,
        "label": "Chronic Kidney Disease Presence Prediction",
        "positive": "chronic kidney disease present (labelled 'ckd')",
        "raw_subdir": "kidney",
        "external_validation": None,
    },
    "diabetes": {
        "uci_id": 891,
        "label": "Diabetes Health-Indicator Risk Prediction",
        "positive": "prediabetes or diabetes reported (BRFSS self-report)",
        "raw_subdir": "diabetes",
        "external_validation": None,
    },
}

# Auxiliary dataset used only for external validation of the heart model.
EXTERNAL_DATASETS: dict[str, dict] = {
    "statlog": {
        "uci_id": 145,
        "label": "Statlog (Heart) — external validation set for the heart model",
        "raw_subdir": "statlog",
    },
}

# --------------------------------------------------------------------------------------
# Risk bands
# --------------------------------------------------------------------------------------
# Presentation buckets for a model-estimated probability. They are NOT clinical decision
# thresholds and carry no medical meaning.
#
# They are anchored on each model's OWN operating threshold rather than on fixed cut-points
# such as 0.33/0.66, because a fixed grid is wrong for any model whose probabilities are
# calibrated to a low-prevalence population. Measured on the diabetes test set (n=50,961)
# with the old fixed bands: 86.8% of respondents landed in "low" and 0.56% in "high", and
# every case the model actually flags at its 0.1388 operating threshold — including every
# true positive it catches — was labelled "low risk". The band contradicted the model.
#
# The upper boundary is the operating threshold, which is a real decision point learned by
# Youden's J on training out-of-fold predictions. The lower boundary is half of it: an
# admittedly arbitrary split of the below-threshold range, kept only so the bucket labels
# have three levels. Only the upper boundary means anything.
RISK_BAND_NOTE: str = (
    "Presentation buckets, not clinical categories. The boundary between 'moderate' and "
    "'high' is this model's own operating threshold (Youden's J on training out-of-fold "
    "predictions), so 'high' means only 'this model would flag this case at its selected "
    "operating point'. The low/moderate boundary is half the threshold — an arbitrary "
    "split of the remaining range with no medical meaning. Neither boundary is a clinical "
    "decision threshold, and the bands say nothing about diagnosis or severity."
)


def risk_bands(threshold: float) -> list[tuple[str, float, float]]:
    """Presentation bands for one model, anchored on its operating ``threshold``."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"operating threshold must be in (0, 1], got {threshold}")
    return [
        ("low", 0.0, threshold / 2.0),
        ("moderate", threshold / 2.0, threshold),
        ("high", threshold, 1.01),
    ]


def risk_band(probability: float, threshold: float) -> str:
    """Map a model-estimated probability to a presentation band label.

    ``threshold`` is the model's operating threshold; there is deliberately no default,
    because a band computed without it is meaningless (see :data:`RISK_BAND_NOTE`).
    """
    for name, lo, hi in risk_bands(threshold):
        if lo <= probability < hi:
            return name
    return "unknown"


# --------------------------------------------------------------------------------------
# Disclaimer (must accompany every prediction, report and model card)
# --------------------------------------------------------------------------------------
DISCLAIMER: str = (
    "For research and educational purposes only. Predictions generated by this system "
    "are based on machine-learning models trained on historical public datasets and "
    "must not be interpreted as medical diagnosis or professional medical advice."
)

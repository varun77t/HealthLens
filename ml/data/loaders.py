"""Disease-specific dataset loaders with local caching.

Each ``load_*`` function returns ``(X, y, spec)`` where:
    X     : pandas DataFrame of predictor columns (raw values, cleaned of encoding
            artefacts but NOT imputed/scaled — preprocessing happens inside the
            model pipeline to avoid leakage)
    y     : pandas Series, integer 0/1 target after the documented transformation
    spec  : FeatureSpec describing column roles and meanings

On first call a dataset is fetched from the UCI ML Repository via ``ucimlrepo`` and
cached as CSV under ``data/raw/<subdir>/``. Subsequent calls read the CSV, so the
pipeline is reproducible and works offline once the cache is populated.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from config import DISEASES, EXTERNAL_DATASETS, RAW_DATA_DIR
from ml.data.feature_spec import FeatureSpec

# ------------------------------------------------------------------------------------
# Caching helpers
# ------------------------------------------------------------------------------------


def _raw_dir(subdir: str) -> Path:
    d = RAW_DATA_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_raw(uci_id: int, subdir: str, *, force: bool = False) -> pd.DataFrame:
    """Return the raw (features + target) frame for a UCI dataset, using a CSV cache.

    Also caches the dataset's variable metadata as ``<subdir>_variables.csv``.
    """
    raw_dir = _raw_dir(subdir)
    data_csv = raw_dir / f"{subdir}.csv"
    vars_csv = raw_dir / f"{subdir}_variables.csv"

    if data_csv.exists() and not force:
        return pd.read_csv(data_csv)

    from ucimlrepo import fetch_ucirepo  # imported lazily so the cache path needs no net

    ds = fetch_ucirepo(id=uci_id)
    features: pd.DataFrame = ds.data.features.copy()
    targets: pd.DataFrame = ds.data.targets.copy()
    raw = pd.concat([features, targets], axis=1)

    raw.to_csv(data_csv, index=False)
    try:
        ds.variables.to_csv(vars_csv, index=False)
    except Exception as exc:  # pragma: no cover - metadata is best-effort
        warnings.warn(f"Could not cache variable metadata for id={uci_id}: {exc}")
    return raw


def _strip_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace/tab artefacts from every object column (UCI CKD is full of them)."""
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].str.strip()
        out[col] = out[col].replace({"?": pd.NA, "": pd.NA, "nan": pd.NA})
    return out


# ------------------------------------------------------------------------------------
# Heart Disease — UCI id 45 (Cleveland processed)
# ------------------------------------------------------------------------------------

_HEART_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
_HEART_CATEGORICAL = ["cp", "restecg", "slope", "ca", "thal"]
_HEART_BINARY = ["sex", "fbs", "exang"]
_HEART_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic)",
    "trestbps": "Resting blood pressure on admission (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting electrocardiographic result (0 normal, 1 ST-T abnormality, 2 LV hypertrophy)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (1 upsloping, 2 flat, 3 downsloping)",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "Thalassemia (3 normal, 6 fixed defect, 7 reversible defect)",
    "num": "Angiographic disease status (0 = <50% narrowing, 1-4 = >50% narrowing)",
}


def load_heart(*, force: bool = False) -> tuple[pd.DataFrame, pd.Series, FeatureSpec]:
    raw = fetch_raw(DISEASES["heart"]["uci_id"], DISEASES["heart"]["raw_subdir"], force=force)
    raw = _strip_object_cols(raw)

    target_col = "num"
    feats = [*_HEART_NUMERIC, *_HEART_CATEGORICAL, *_HEART_BINARY]
    X = raw[feats].apply(pd.to_numeric, errors="coerce")
    # Target transform: 0 = absence, any positive angiographic grade = presence.
    y = (pd.to_numeric(raw[target_col], errors="coerce") > 0).astype("int64")
    y.name = "heart_disease_present"

    spec = FeatureSpec(
        disease="heart",
        target="heart_disease_present",
        numeric=_HEART_NUMERIC,
        categorical=_HEART_CATEGORICAL,
        binary=_HEART_BINARY,
        descriptions=_HEART_DESCRIPTIONS,
        notes=(
            "Target: raw 'num' in {0..4} mapped to binary (num > 0 => 1). "
            "'ca' and 'thal' contain missing values (originally '?'), imputed inside "
            "the model pipeline. 'ca' (vessel count) and 'slope' are ordinal but "
            "modelled as categorical. Cleveland single-site cohort, n=303."
        ),
    )
    return X, y, spec


# ------------------------------------------------------------------------------------
# Chronic Kidney Disease — UCI id 336
# ------------------------------------------------------------------------------------

_KIDNEY_NUMERIC = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
]
_KIDNEY_BINARY = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
_KIDNEY_DESCRIPTIONS = {
    "age": "Age in years",
    "bp": "Blood pressure (mm Hg)",
    "sg": "Urine specific gravity (1.005-1.025)",
    "al": "Albumin (0-5 ordinal scale)",
    "su": "Sugar (0-5 ordinal scale)",
    "rbc": "Red blood cells in urine (normal / abnormal)",
    "pc": "Pus cell (normal / abnormal)",
    "pcc": "Pus cell clumps (present / notpresent)",
    "ba": "Bacteria (present / notpresent)",
    "bgr": "Random blood glucose (mg/dl)",
    "bu": "Blood urea (mg/dl)",
    "sc": "Serum creatinine (mg/dl)",
    "sod": "Sodium (mEq/L)",
    "pot": "Potassium (mEq/L)",
    "hemo": "Haemoglobin (g/dl)",
    "pcv": "Packed cell volume (%)",
    "wbcc": "White blood cell count (cells/cu mm)",
    "rbcc": "Red blood cell count (millions/cu mm)",
    "htn": "Hypertension (yes / no)",
    "dm": "Diabetes mellitus (yes / no)",
    "cad": "Coronary artery disease (yes / no)",
    "appet": "Appetite (good / poor)",
    "pe": "Pedal edema (yes / no)",
    "ane": "Anaemia (yes / no)",
    "class": "Diagnosis class (ckd / notckd)",
}
# Categorical text values mapped to 0/1 so the whole kidney feature set is numeric-binary.
_KIDNEY_BINARY_MAP = {
    "rbc": {"normal": 0, "abnormal": 1},
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1},
    "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1},
    "dm": {"no": 0, "yes": 1},
    "cad": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe": {"no": 0, "yes": 1},
    "ane": {"no": 0, "yes": 1},
}


def load_kidney(*, force: bool = False) -> tuple[pd.DataFrame, pd.Series, FeatureSpec]:
    raw = fetch_raw(DISEASES["kidney"]["uci_id"], DISEASES["kidney"]["raw_subdir"], force=force)
    raw = _strip_object_cols(raw)

    # Some numeric columns (pcv, wc, rc) arrive as strings because of stray tabs.
    num = raw[_KIDNEY_NUMERIC].apply(pd.to_numeric, errors="coerce")

    bin_frames: dict[str, pd.Series] = {}
    for col in _KIDNEY_BINARY:
        series = raw[col]
        if series.dtype == object:
            mapped = series.str.lower().map(_KIDNEY_BINARY_MAP[col])
        else:
            mapped = series
        bin_frames[col] = pd.to_numeric(mapped, errors="coerce")
    binary_df = pd.DataFrame(bin_frames)

    X = pd.concat([num, binary_df], axis=1)[_KIDNEY_NUMERIC + _KIDNEY_BINARY]

    cls = raw["class"].astype("string").str.lower().str.strip()
    y = (cls == "ckd").astype("int64")
    y.name = "ckd_present"

    spec = FeatureSpec(
        disease="kidney",
        target="ckd_present",
        numeric=_KIDNEY_NUMERIC,
        categorical=[],
        binary=_KIDNEY_BINARY,
        descriptions=_KIDNEY_DESCRIPTIONS,
        notes=(
            "Target: 'class' text {'ckd','notckd'} mapped to {1,0}. Extensive missing "
            "values across most columns; imputed inside the model pipeline. Numeric "
            "columns pcv/wbcc/rbcc were stored as text (stray tab characters) and coerced "
            "to numbers. Nominal yes/no and normal/abnormal columns mapped to 0/1. "
            "'sg','al','su' are ordinal clinical scales kept as numeric. n=400."
        ),
    )
    return X, y, spec


# ------------------------------------------------------------------------------------
# CDC Diabetes Health Indicators — UCI id 891
# ------------------------------------------------------------------------------------

_DIA_NUMERIC = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income"]
_DIA_BINARY = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk", "Sex",
]
_DIA_DESCRIPTIONS = {
    "Diabetes_binary": "0 = no diabetes, 1 = prediabetes or diabetes (BRFSS self-report)",
    "HighBP": "Ever told by a health professional you have high blood pressure (0/1)",
    "HighChol": "Ever told you have high cholesterol (0/1)",
    "CholCheck": "Cholesterol check within the past 5 years (0/1)",
    "BMI": "Body Mass Index",
    "Smoker": "Smoked at least 100 cigarettes in lifetime (0/1)",
    "Stroke": "Ever told you had a stroke (0/1)",
    "HeartDiseaseorAttack": "Coronary heart disease or myocardial infarction (0/1)",
    "PhysActivity": "Physical activity in past 30 days, not job-related (0/1)",
    "Fruits": "Consumes fruit 1+ times per day (0/1)",
    "Veggies": "Consumes vegetables 1+ times per day (0/1)",
    "HvyAlcoholConsump": "Heavy alcohol consumption (0/1)",
    "AnyHealthcare": "Has any health-care coverage (0/1)",
    "NoDocbcCost": "Could not see a doctor in past 12 months because of cost (0/1)",
    "GenHlth": "Self-rated general health (1 excellent - 5 poor)",
    "MentHlth": "Days of poor mental health in past 30 days (0-30)",
    "PhysHlth": "Days of poor physical health in past 30 days (0-30)",
    "DiffWalk": "Serious difficulty walking or climbing stairs (0/1)",
    "Sex": "Sex (0 = female, 1 = male)",
    "Age": "13-level age category (1 = 18-24 ... 13 = 80+)",
    "Education": "Education level (1 - 6 ordinal)",
    "Income": "Income category (1 - 8 ordinal)",
}


def load_diabetes(*, force: bool = False) -> tuple[pd.DataFrame, pd.Series, FeatureSpec]:
    raw = fetch_raw(DISEASES["diabetes"]["uci_id"], DISEASES["diabetes"]["raw_subdir"], force=force)

    target_col = "Diabetes_binary"
    feats = [*_DIA_NUMERIC, *_DIA_BINARY]
    X = raw[feats].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(raw[target_col], errors="coerce").astype("int64")
    y.name = "diabetes_or_prediabetes"

    spec = FeatureSpec(
        disease="diabetes",
        target="diabetes_or_prediabetes",
        numeric=_DIA_NUMERIC,
        categorical=[],
        binary=_DIA_BINARY,
        descriptions=_DIA_DESCRIPTIONS,
        notes=(
            "Target: 'Diabetes_binary' where 1 = prediabetes OR diabetes, 0 = neither. "
            "Derived from the CDC BRFSS survey (self-reported). Features are health "
            "indicators, not laboratory measurements: this module estimates a RISK "
            "ASSOCIATION, not a diagnosis. Full imbalanced release (~14% positive), "
            "n=253,680. Ordinal survey codes (GenHlth, Age, Education, Income) are "
            "modelled as numeric."
        ),
    )
    return X, y, spec


# ------------------------------------------------------------------------------------
# Statlog (Heart) — UCI id 145 — external validation only
# ------------------------------------------------------------------------------------

_STATLOG_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def load_statlog(*, force: bool = False) -> tuple[pd.DataFrame, pd.Series, FeatureSpec]:
    """Statlog Heart, aligned to the Cleveland (id 45) feature schema for external eval."""
    raw = fetch_raw(
        EXTERNAL_DATASETS["statlog"]["uci_id"],
        EXTERNAL_DATASETS["statlog"]["raw_subdir"],
        force=force,
    )
    # ucimlrepo returns generic attribute names; the column order matches Cleveland's 13.
    feature_cols = list(raw.columns[:13])
    target_col = raw.columns[13]
    X = raw[feature_cols].copy()
    X.columns = _STATLOG_COLUMNS
    X = X.apply(pd.to_numeric, errors="coerce")
    # Reorder to the shared heart schema (numeric -> categorical -> binary).
    X = X[[*_HEART_NUMERIC, *_HEART_CATEGORICAL, *_HEART_BINARY]]

    # Statlog 'thal' is coded 3/6/7 like Cleveland; 'slope'/'cp' also share coding.
    y_raw = pd.to_numeric(raw[target_col], errors="coerce")
    # Statlog target: 1 = absence, 2 = presence  ->  0 / 1
    y = (y_raw == 2).astype("int64")
    y.name = "heart_disease_present"

    spec = FeatureSpec(
        disease="statlog",
        target="heart_disease_present",
        numeric=_HEART_NUMERIC,
        categorical=_HEART_CATEGORICAL,
        binary=_HEART_BINARY,
        descriptions=_HEART_DESCRIPTIONS,
        notes=(
            "External validation set for the heart model. Target 1/2 mapped to 0/1. "
            "Columns renamed positionally to the Cleveland schema; encodings for "
            "cp/slope/thal are believed compatible. n=270, no missing values."
        ),
    )
    return X, y, spec


LOADERS = {
    "heart": load_heart,
    "kidney": load_kidney,
    "diabetes": load_diabetes,
    "statlog": load_statlog,
}


def load(disease: str, *, force: bool = False):
    if disease not in LOADERS:
        raise KeyError(f"Unknown dataset '{disease}'. Options: {sorted(LOADERS)}")
    return LOADERS[disease](force=force)

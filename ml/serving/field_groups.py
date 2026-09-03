"""Presentation metadata for input fields: where a person finds each value, and what it means.

The model pipelines think in columns (``sg``, ``thal``, ``GenHlth``). A person filling a form
thinks in documents: *the urine test I'm holding*, *the stress-test report*, *questions about
me*. This module carries that translation, plus the labels that turn ``sex: 1`` into "Male"
and ``appet: 1`` into "Poor".

None of it is derived from the data, so none of it is generated — it is domain knowledge about
where a measurement comes from, hand-authored and reviewed. It is kept out of
:class:`~ml.data.feature_spec.FeatureSpec` because the training pipeline has no use for it.

Provenance of the coded value labels is recorded in :data:`LABEL_SOURCES`. Where the cached UCI
variable metadata states a coding explicitly it is used verbatim; where it gives only anchor
levels, the source note says so rather than presenting the rest as verified.
"""
from __future__ import annotations

# ------------------------------------------------------------------------------------
# Groups — ordered by how a person would work through them
# ------------------------------------------------------------------------------------

GROUPS: dict[str, dict] = {
    "about_you": {
        "label": "About you",
        "help": "Basic details. No report needed.",
        "order": 1,
    },
    "vitals": {
        "label": "Vitals and measurements",
        "help": "Blood pressure, height and weight — from a clinic visit or measured at home.",
        "order": 2,
    },
    "symptoms": {
        "label": "Symptoms and clinical signs",
        "help": "What you or a clinician have observed. No report needed.",
        "order": 3,
    },
    "history": {
        "label": "Your medical history",
        "help": "Conditions you have previously been told you have.",
        "order": 4,
    },
    "blood_test": {
        "label": "From your blood test",
        "help": "A routine blood panel — full blood count, metabolic panel, or lipid profile.",
        "order": 5,
    },
    "urine_test": {
        "label": "From your urine test",
        "help": "A standard urinalysis report.",
        "order": 6,
    },
    "resting_ecg": {
        "label": "From a resting ECG",
        "help": "A standard 12-lead electrocardiogram taken at rest.",
        "order": 7,
    },
    "exercise_test": {
        "label": "From an exercise stress test",
        "help": "A treadmill or bicycle ECG stress test report.",
        "order": 8,
    },
    "cardiac_imaging": {
        "label": "From cardiac imaging",
        "help": "A coronary angiogram (fluoroscopy) and a thallium perfusion scan. These are "
                "specialist investigations — if you have not had them, say so, and the "
                "estimate will be reported as reduced-coverage rather than quietly guessed.",
        "order": 9,
    },
    "wellbeing": {
        "label": "General wellbeing",
        "help": "How you would describe your own health recently.",
        "order": 10,
    },
    "lifestyle": {
        "label": "Lifestyle",
        "help": "Everyday habits.",
        "order": 11,
    },
    "healthcare_access": {
        "label": "Healthcare access",
        "help": "Your access to care. Asked because the training survey asked it.",
        "order": 12,
    },
}

# ------------------------------------------------------------------------------------
# field -> group
# ------------------------------------------------------------------------------------

FIELD_GROUPS: dict[str, dict[str, str]] = {
    "heart": {
        "age": "about_you", "sex": "about_you",
        "trestbps": "vitals",
        "cp": "symptoms",
        "chol": "blood_test", "fbs": "blood_test",
        "restecg": "resting_ecg",
        "thalach": "exercise_test", "exang": "exercise_test",
        "oldpeak": "exercise_test", "slope": "exercise_test",
        "ca": "cardiac_imaging", "thal": "cardiac_imaging",
    },
    "kidney": {
        "age": "about_you",
        "bp": "vitals",
        "appet": "symptoms", "pe": "symptoms", "ane": "symptoms",
        "htn": "history", "dm": "history", "cad": "history",
        "bgr": "blood_test", "bu": "blood_test", "sc": "blood_test",
        "sod": "blood_test", "pot": "blood_test", "hemo": "blood_test",
        "pcv": "blood_test", "wbcc": "blood_test", "rbcc": "blood_test",
        "sg": "urine_test", "al": "urine_test", "su": "urine_test",
        "rbc": "urine_test", "pc": "urine_test", "pcc": "urine_test", "ba": "urine_test",
    },
    "diabetes": {
        "Age": "about_you", "Sex": "about_you",
        "Education": "about_you", "Income": "about_you",
        "BMI": "vitals",
        "DiffWalk": "symptoms",
        "HighBP": "history", "HighChol": "history", "CholCheck": "history",
        "Stroke": "history", "HeartDiseaseorAttack": "history",
        "GenHlth": "wellbeing", "MentHlth": "wellbeing", "PhysHlth": "wellbeing",
        "Smoker": "lifestyle", "PhysActivity": "lifestyle", "Fruits": "lifestyle",
        "Veggies": "lifestyle", "HvyAlcoholConsump": "lifestyle",
        "AnyHealthcare": "healthcare_access", "NoDocbcCost": "healthcare_access",
    },
}

# ------------------------------------------------------------------------------------
# Short field labels — what the form shows above the input
# ------------------------------------------------------------------------------------

FIELD_LABELS: dict[str, dict[str, str]] = {
    "heart": {
        "age": "Age",
        "sex": "Sex",
        "trestbps": "Resting blood pressure (systolic)",
        "chol": "Serum cholesterol",
        "fbs": "Fasting blood sugar above 120 mg/dl",
        "restecg": "Resting ECG result",
        "thalach": "Maximum heart rate achieved",
        "exang": "Angina brought on by exercise",
        "oldpeak": "ST depression, exercise vs rest",
        "slope": "Slope of the peak exercise ST segment",
        "cp": "Chest pain type",
        "ca": "Major vessels seen on fluoroscopy",
        "thal": "Thallium scan result",
    },
    "kidney": {
        "age": "Age",
        "bp": "Blood pressure (diastolic)",
        "sg": "Urine specific gravity",
        "al": "Albumin in urine",
        "su": "Sugar in urine",
        "rbc": "Red blood cells in urine",
        "pc": "Pus cells",
        "pcc": "Pus cell clumps",
        "ba": "Bacteria",
        "bgr": "Random blood glucose",
        "bu": "Blood urea",
        "sc": "Serum creatinine",
        "sod": "Sodium",
        "pot": "Potassium",
        "hemo": "Haemoglobin",
        "pcv": "Packed cell volume",
        "wbcc": "White blood cell count",
        "rbcc": "Red blood cell count",
        "htn": "Diagnosed with high blood pressure",
        "dm": "Diagnosed with diabetes",
        "cad": "Diagnosed with coronary artery disease",
        "appet": "Appetite",
        "pe": "Swelling in feet or ankles",
        "ane": "Diagnosed with anaemia",
    },
    "diabetes": {
        "Age": "Age group",
        "Sex": "Sex",
        "Education": "Highest level of education",
        "Income": "Annual household income",
        "BMI": "Body Mass Index",
        "HighBP": "Ever told you have high blood pressure",
        "HighChol": "Ever told you have high cholesterol",
        "CholCheck": "Cholesterol checked in the last 5 years",
        "Stroke": "Ever told you had a stroke",
        "HeartDiseaseorAttack": "Ever told you had coronary heart disease or a heart attack",
        "GenHlth": "In general, your health is",
        "MentHlth": "Days of poor mental health in the last 30",
        "PhysHlth": "Days of poor physical health in the last 30",
        "DiffWalk": "Serious difficulty walking or climbing stairs",
        "Smoker": "Smoked at least 100 cigarettes in your life",
        "PhysActivity": "Physical activity in the last 30 days, not job-related",
        "Fruits": "Eat fruit at least once a day",
        "Veggies": "Eat vegetables at least once a day",
        "HvyAlcoholConsump": "Heavy alcohol consumption",
        "AnyHealthcare": "Have any health-care coverage",
        "NoDocbcCost": "Could not see a doctor in the last year because of cost",
    },
}

# ------------------------------------------------------------------------------------
# Coded value labels — so a form never shows a person a bare 0/1/7
# ------------------------------------------------------------------------------------

_YES_NO = {"0": "No", "1": "Yes"}

VALUE_LABELS: dict[str, dict[str, dict[str, str]]] = {
    "heart": {
        "sex": {"0": "Female", "1": "Male"},
        "cp": {
            "1": "Typical angina",
            "2": "Atypical angina",
            "3": "Non-anginal pain",
            "4": "No chest pain (asymptomatic)",
        },
        "fbs": {"0": "No — 120 mg/dl or below", "1": "Yes — above 120 mg/dl"},
        "restecg": {
            "0": "Normal",
            "1": "ST-T wave abnormality",
            "2": "Probable or definite left ventricular hypertrophy",
        },
        "exang": _YES_NO,
        "slope": {"1": "Upsloping", "2": "Flat", "3": "Downsloping"},
        "ca": {"0": "0 vessels", "1": "1 vessel", "2": "2 vessels", "3": "3 vessels"},
        "thal": {"3": "Normal", "6": "Fixed defect", "7": "Reversible defect"},
    },
    "kidney": {
        "rbc": {"0": "Normal", "1": "Abnormal"},
        "pc": {"0": "Normal", "1": "Abnormal"},
        "pcc": {"0": "Not present", "1": "Present"},
        "ba": {"0": "Not present", "1": "Present"},
        "htn": _YES_NO,
        "dm": _YES_NO,
        "cad": _YES_NO,
        "appet": {"0": "Good", "1": "Poor"},
        "pe": _YES_NO,
        "ane": _YES_NO,
    },
    "diabetes": {
        "Sex": {"0": "Female", "1": "Male"},
        "GenHlth": {
            "1": "Excellent", "2": "Very good", "3": "Good", "4": "Fair", "5": "Poor",
        },
        "Age": {
            "1": "18-24", "2": "25-29", "3": "30-34", "4": "35-39", "5": "40-44",
            "6": "45-49", "7": "50-54", "8": "55-59", "9": "60-64", "10": "65-69",
            "11": "70-74", "12": "75-79", "13": "80 or older",
        },
        "Education": {
            "1": "Never attended school or only kindergarten",
            "2": "Grades 1-8 (elementary)",
            "3": "Grades 9-11 (some high school)",
            "4": "Grade 12 or GED (high school graduate)",
            "5": "College 1-3 years (some college or technical school)",
            "6": "College 4 years or more (college graduate)",
        },
        "Income": {
            "1": "Less than $10,000", "2": "Less than $15,000",
            "3": "Less than $20,000", "4": "Less than $25,000",
            "5": "Less than $35,000", "6": "Less than $50,000",
            "7": "Less than $75,000", "8": "$75,000 or more",
        },
        "HighBP": _YES_NO,
        "HighChol": _YES_NO,
        "CholCheck": _YES_NO,
        "Smoker": _YES_NO,
        "Stroke": _YES_NO,
        "HeartDiseaseorAttack": _YES_NO,
        "PhysActivity": _YES_NO,
        "Fruits": _YES_NO,
        "Veggies": _YES_NO,
        "HvyAlcoholConsump": _YES_NO,
        "AnyHealthcare": _YES_NO,
        "NoDocbcCost": _YES_NO,
        "DiffWalk": _YES_NO,
    },
}

# ------------------------------------------------------------------------------------
# Units
# ------------------------------------------------------------------------------------
# Units were previously only readable inside each feature's prose description, which is
# fine for an API and useless for a UI that wants to print "12.4 g/dL". They are the units
# the model was TRAINED on: a value entered or extracted in anything else is wrong, not
# merely differently formatted. Coded and dimensionless fields carry no unit.
UNITS: dict[str, dict[str, str]] = {
    "heart": {
        "age": "years",
        "trestbps": "mm Hg",
        "chol": "mg/dL",
        "thalach": "bpm",
        "oldpeak": "mm",
    },
    "kidney": {
        "age": "years",
        "bp": "mm Hg",
        "bgr": "mg/dL",
        "bu": "mg/dL",
        "sc": "mg/dL",
        "sod": "mEq/L",
        "pot": "mEq/L",
        "hemo": "g/dL",
        "pcv": "%",
        "wbcc": "cells/cumm",
        "rbcc": "millions/cmm",
        # sg (specific gravity), al and su (0-5 ordinal scales) are dimensionless.
    },
    "diabetes": {
        "BMI": "kg/m²",
        "MentHlth": "days",
        "PhysHlth": "days",
    },
}


def unit_of(disease: str, field: str) -> str:
    """The training unit for ``field``, or an empty string if it is coded/dimensionless."""
    return UNITS.get(disease, {}).get(field, "")


LABEL_SOURCES: dict[str, str] = {
    "heart": "Cleveland processed release (UCI id 45) attribute documentation.",
    "kidney": "UCI Chronic Kidney Disease (id 336) attribute documentation. The loader maps "
              "each nominal text value to 0/1; these labels restate that mapping.",
    "diabetes": "Cached UCI variable metadata (data/raw/diabetes/diabetes_variables.csv) "
                "states the GenHlth and Education codings in full, and gives anchor levels "
                "for Age (1 = 18-24, 9 = 60-64, 13 = 80 or older) and Income "
                "(1 = <$10,000, 5 = <$35,000, 8 = $75,000+). The Age labels follow from "
                "those anchors, which fix the five-year banding. The intermediate Income "
                "levels (2, 3, 4, 6, 7) are NOT stated in that metadata; they follow the "
                "BRFSS INCOME2 codebook and are consistent with all three anchors, but are "
                "not independently verified here.",
}


def groups_for(disease: str) -> list[dict]:
    """Ordered group definitions actually used by ``disease``."""
    used = set(FIELD_GROUPS.get(disease, {}).values())
    out = [{"id": gid, **GROUPS[gid]} for gid in GROUPS if gid in used]
    return sorted(out, key=lambda g: g["order"])


def group_of(disease: str, field: str) -> str:
    if field not in FIELD_GROUPS.get(disease, {}):
        raise KeyError(
            f"No presentation group recorded for '{disease}.{field}'. Add it to "
            f"FIELD_GROUPS — an ungrouped field would not appear on the form at all."
        )
    return FIELD_GROUPS[disease][field]


def field_label(disease: str, field: str, fallback: str = "") -> str:
    return FIELD_LABELS.get(disease, {}).get(field) or fallback or field


def options_for(
    disease: str,
    field: str,
    categories: list[float] | None,
    *,
    observed: tuple[float, float] | None = None,
) -> list[dict] | None:
    """``[{value, label}]`` for a field a person should pick from, else ``None``.

    Two cases produce options:

    * a genuine categorical/binary column, whose ``categories`` come from the data;
    * an **ordinal survey code modelled as numeric** — ``Age``, ``GenHlth``, ``Education``
      and ``Income`` are 1-13 / 1-5 / 1-6 / 1-8 scales that the pipeline treats as numbers.
      Those must still be *chosen*, not typed: asking someone to enter ``8`` for their age
      is how a form gets a real age typed into a band code. The value sent to the model is
      unchanged, so this is presentation only.

    For the ordinal case every level across the observed training range must be labelled,
    otherwise a value the model was trained on would be unselectable. That is checked here
    and raised rather than silently dropped.
    """
    labels = VALUE_LABELS.get(disease, {}).get(field, {})

    if categories:
        # A category in the data but missing from the table keeps its raw value as a label
        # rather than vanishing from the form.
        return [{"value": c, "label": labels.get(f"{c:g}", f"{c:g}")} for c in categories]

    if not labels or observed is None:
        return None

    lo, hi = observed
    if lo != int(lo) or hi != int(hi):
        return None  # a continuous quantity that happens to share a name with a code
    levels = [float(v) for v in range(int(lo), int(hi) + 1)]
    unlabelled = [f"{v:g}" for v in levels if f"{v:g}" not in labels]
    if unlabelled:
        raise KeyError(
            f"'{disease}.{field}' is an ordinal code observed over {lo:g}-{hi:g}, but "
            f"levels {', '.join(unlabelled)} have no label in VALUE_LABELS. Add them — an "
            f"unlabelled level would be unselectable on the form despite appearing in the "
            f"training data."
        )
    return [{"value": v, "label": labels[f"{v:g}"]} for v in levels]

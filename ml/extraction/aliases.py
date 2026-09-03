"""Alternative wordings for field labels, and how a printed blood-pressure pair is split.

The demonstration reports print each field's canonical label from ``serving.json``, so exact
matching would be enough for them. These aliases exist so the extractor is not *only* able to
read documents this project generated: real reports say "Hemoglobin", "PCV", "BUN" or
"Creatinine" rather than the exact strings in the model's schema.

Nothing here maps to a value — only to a *field name*. Values are resolved against each
field's own option list from ``serving.json``, so a coded field can never acquire a level the
model was not trained on.
"""
from __future__ import annotations

# field name -> additional strings that should resolve to it. Matching is
# case-insensitive and punctuation-insensitive; the canonical label is added automatically.
ALIASES: dict[str, dict[str, list[str]]] = {
    "kidney": {
        "age": ["age", "patient age"],
        "bp": ["blood pressure", "bp", "diastolic blood pressure", "bp (diastolic)"],
        "sg": ["specific gravity", "urine sg", "sg"],
        "al": ["albumin", "urine albumin", "albuminuria", "al"],
        "su": ["sugar", "urine sugar", "glycosuria", "su"],
        "rbc": ["red blood cells", "rbc", "urine rbc", "red blood cells (urine)"],
        "pc": ["pus cell", "pus cells", "pc"],
        "pcc": ["pus cell clumps", "pcc"],
        "ba": ["bacteria", "ba"],
        "bgr": ["blood glucose random", "random blood sugar", "rbs", "bgr",
                "blood glucose", "glucose (random)"],
        "bu": ["blood urea", "urea", "bun", "blood urea nitrogen", "bu"],
        "sc": ["serum creatinine", "creatinine", "sc"],
        "sod": ["sodium", "na", "serum sodium", "sod"],
        "pot": ["potassium", "k", "serum potassium", "pot"],
        "hemo": ["hemoglobin", "haemoglobin", "hb", "hgb", "hemo"],
        "pcv": ["packed cell volume", "pcv", "hematocrit", "haematocrit", "hct"],
        "wbcc": ["white blood cell count", "wbc count", "wbc", "wc",
                 "total leucocyte count", "tlc", "wbcc"],
        "rbcc": ["red blood cell count", "rbc count", "rc", "rbcc"],
        "htn": ["hypertension", "htn", "high blood pressure history"],
        "dm": ["diabetes mellitus", "dm", "diabetes"],
        "cad": ["coronary artery disease", "cad"],
        "appet": ["appetite", "appet"],
        "pe": ["pedal edema", "pedal oedema", "swelling", "pe"],
        "ane": ["anemia", "anaemia", "ane"],
    },
    "heart": {
        "age": ["age", "patient age"],
        "sex": ["sex", "gender"],
        "cp": ["chest pain", "chest pain type", "cp"],
        "trestbps": ["resting blood pressure", "blood pressure", "systolic blood pressure",
                     "bp", "trestbps"],
        "chol": ["serum cholesterol", "cholesterol", "total cholesterol", "chol"],
        "fbs": ["fasting blood sugar", "fasting glucose", "fbs"],
        "restecg": ["resting ecg", "resting electrocardiogram", "ecg", "restecg"],
        "thalach": ["maximum heart rate", "max heart rate", "peak heart rate", "thalach"],
        "exang": ["exercise induced angina", "exercise angina", "exang"],
        "oldpeak": ["st depression", "oldpeak"],
        "slope": ["st segment slope", "slope of peak exercise st segment", "slope"],
        "ca": ["major vessels", "vessels coloured by fluoroscopy",
               "vessels colored by fluoroscopy", "ca"],
        "thal": ["thallium scan", "thallium", "thal", "perfusion scan"],
    },
    "diabetes": {
        "BMI": ["body mass index", "bmi"],
        "Age": ["age group", "age band", "age"],
        "Sex": ["sex", "gender"],
        "GenHlth": ["general health", "self rated health", "in general your health is"],
        "MentHlth": ["poor mental health days", "mental health days"],
        "PhysHlth": ["poor physical health days", "physical health days"],
        "HighBP": ["high blood pressure", "hypertension"],
        "HighChol": ["high cholesterol"],
        "CholCheck": ["cholesterol check", "cholesterol checked"],
        "Smoker": ["smoker", "smoking"],
        "Stroke": ["stroke"],
        "HeartDiseaseorAttack": ["heart disease", "coronary heart disease", "heart attack"],
        "PhysActivity": ["physical activity"],
        "Fruits": ["fruit", "fruits"],
        "Veggies": ["vegetables", "veggies"],
        "HvyAlcoholConsump": ["heavy alcohol consumption", "alcohol"],
        "AnyHealthcare": ["health care coverage", "healthcare coverage", "insurance"],
        "NoDocbcCost": ["could not see a doctor because of cost", "cost barrier to care"],
        "DiffWalk": ["difficulty walking", "difficulty walking or climbing stairs"],
        "Education": ["education", "education level"],
        "Income": ["income", "household income", "annual household income"],
    },
}

# Text a report uses to say a test was not carried out. These become an explicit `missing`,
# never a guessed value.
NOT_MEASURED_TEXTS = {
    "not performed", "not done", "not measured", "not recorded", "not available",
    "n/a", "na", "nil", "-", "--", "not tested", "pending",
}

# Where a report prints a blood pressure as one "142/90" pair, which half each model wants.
#
# This is not cosmetic. Heart's `trestbps` is systolic (training range 94-200, median 130)
# and kidney's `bp` is diastolic (range 50-180, median 80). Feeding a systolic reading into
# the kidney model would be silently, confidently wrong, so the split is explicit per field
# rather than inferred from the label.
BP_COMPONENT: dict[str, str] = {
    "heart.trestbps": "systolic",
    "kidney.bp": "diastolic",
}

# Project Status — Multi-Disease AI

_Last updated: 2026-09-02_

Research/education platform. **Not** a clinical diagnostic tool. Three independent
disease pipelines (heart, kidney, diabetes). Scope of current build effort: through the
FastAPI backend (Phases 0–8 of the implementation plan); React frontend and CNN/Grad-CAM
module deferred.

Plan file: `C:\Users\Admin\.claude\plans\you-are-working-on-zippy-hammock.md`

## Phase progress

| Phase | Scope | State |
|---|---|---|
| 0 | Repo scaffold, environment, `config.py` | ✅ complete (`f64caa9`, `e3a06b4`) |
| 1 | Dataset acquisition, caching, loaders, EDA, target definitions | ✅ complete (`f64caa9`, `e3a06b4`) |
| 2 | Shared leakage-safe preprocessing + training/evaluation package (no training) | ⬜ next |
| 3–5 | Heart / Kidney / Diabetes models end-to-end (CV, tuning, SHAP, model cards) | ⬜ |
| 6 | External validation (Heart → Statlog) | ⬜ |
| 7 | Calibration + fairness analysis | ⬜ |
| 8 | FastAPI backend (`/predict/*`, `/models`, `/analytics/*`, `/scenario/*`) | ⬜ |
| — | React frontend, CNN/Grad-CAM imaging | deferred |

**No models trained. No performance metrics exist yet — none are fabricated anywhere in the repo.**

## Environment

- Python 3.12.6. Project runs in its own venv: `.venv/` (git-ignored).
- venv key versions: numpy 2.5.2, pandas 2.3.0, scikit-learn 1.7.2, xgboost 3.0.5,
  lightgbm 4.6.0, imbalanced-learn 0.14.0, shap 0.52.0, numba 0.67.0, ucimlrepo 0.0.7,
  matplotlib 3.10.3, seaborn 0.13.2, fastapi 0.115.0, uvicorn 0.30.6, pydantic 2.9.1.
  Full pinned list in `requirements.txt`.
- Jupyter kernel `medicl` registered → `.venv\Scripts\python.exe`. Use it for the notebooks.
- Global Python restored to numpy 1.26.4 (shap/numba/llvmlite removed from global) so the
  user's global TensorFlow / mediapipe keep working. `shap` requires numpy ≥ 2.1, which is
  why the project is isolated in a venv.

## Phase 1 results (computed, from cached raw data)

### Datasets (loaded + target-transformed)

| dataset | UCI id | rows | features | target (positive = 1) | class 0 / 1 | positive rate | imbalance |
|---|---|---|---|---|---|---|---|
| heart | 45 | 303 | 13 (5 num / 5 cat / 3 bin) | `num > 0` (angiographic disease present) | 164 / 139 | 0.4587 | 1.18 |
| kidney | 336 | 400 | 24 (14 num / 0 cat / 10 bin) | `class == 'ckd'` | 150 / 250 | 0.6250 | 1.67 |
| diabetes | 891 | 253,680 | 21 (7 num / 0 cat / 14 bin) | `Diabetes_binary` (prediabetes or diabetes) | 218,334 / 35,346 | 0.1393 | 6.18 |
| statlog (external-val only) | 145 | 270 | 13 | `target == 2` | 150 / 120 | 0.4444 | 1.25 |

Target transformations and feature dictionaries: `reports/<disease>/TARGET.md`.

### Missing values

- **heart**: `ca` 4 (1.3%), `thal` 2 (0.7%). 6 rows affected.
- **kidney**: every column has some; 1,012 cells; 242/400 rows (60.5%) affected.
  Heaviest — `rbc` 38.0%, `rbcc` 32.8%, `wbcc` 26.5%, `pot` 22.0%, `sod` 21.8%, `pcv` 17.8%.
- **diabetes**: none.
- **statlog**: none.

No imputation at load time — deferred to the in-pipeline preprocessing (Phase 2) to avoid leakage.

### Other EDA findings

- **diabetes has 25,772 duplicate rows** (identical survey indicator profiles; legitimate,
  not dropped). Flag for Phase 2: a naive random split can place identical feature vectors
  in both train and test — decide on split strategy / dedup-for-split.
- heart / kidney / statlog: 0 duplicate rows.
- Strongest numeric–target correlations (descriptive only, not a model):
  heart → oldpeak +0.43, thalach −0.42; kidney → hemo −0.77, pcv −0.74, sg −0.73;
  diabetes → GenHlth +0.29, BMI +0.22.
- IQR outlier scan is diagnostic only — nothing removed (medical extremes are legitimate;
  e.g. kidney `sc` max 76, `bu` max 391).

### Artifacts generated

- `reports/<disease>/`: `TARGET.md`, `EDA.md`, `eda_profile.json`, `missing_values.csv`,
  `numeric_summary.csv`, `correlation_matrix.csv`, `figures/` (6 PNGs each).
- `notebooks/{heart,kidney,diabetes}_eda.ipynb` — thin, executed with outputs.
- `data/raw/<dataset>/` — cached `<name>.csv` + `<name>_variables.csv` (git-ignored).

## Verification status (all from venv)

| step | command | result |
|---|---|---|
| dataset acquisition + validation | `python -m ml.data.download_all` | exit 0, all 4 validate OK |
| target docs | `python -m ml.data.target_docs` | exit 0 |
| EDA | `python -m ml.eda.run` | exit 0 |
| notebooks | `jupyter nbconvert --execute` ×3 | all execute end-to-end |
| tests | `python -m pytest -q` | **11 passed** |

Deterministic artifacts (`EDA.md`, `eda_profile.json`, `*.csv`, `TARGET.md`) regenerate
byte-identical. PNG figures / notebook exec-metadata are not byte-stable across environments
(image encoder differences) — cosmetic only, no data impact.

## Known issues / notes

1. **Global env pre-existing conflicts** (not caused by this project, unchanged):
   `opencv-python 4.12` requires numpy ≥ 2 (contradicts numpy 1.26.4 and mediapipe's numpy < 2);
   `jax/jaxlib` ↔ `ml_dtypes 0.4.1`; `mediapipe` ↔ `protobuf 5.29.5`.
2. Dev packages added to global and left there (numpy-agnostic, harmless):
   `jupyter`, `jupyterlab`, `notebook`, `ipykernel`, `nbconvert`, `pytest`, `ucimlrepo`.
3. Benign warning: Jupyter "kernel running over TCP without encryption" during headless
   notebook execution. No errors anywhere.
4. Statlog feature encodings (cp/slope/thal) are *assumed* compatible with Cleveland for
   Phase 6 external validation — to be verified when that phase runs.

## Next step — Phase 2 (no model training)

Build the shared, leakage-safe ML package in `ml/`:

- `ml/preprocessing/build_preprocessor.py` — per-disease `ColumnTransformer`
  (median impute + scale numeric; most-frequent impute + one-hot categorical), always used
  inside a `Pipeline`; never fit on the full dataset.
- `ml/training/model_zoo.py` — LogisticRegression, RandomForest, XGBoost, SVC, LightGBM,
  each wrapped as `imblearn.pipeline.Pipeline` with optional in-fold SMOTE.
- `ml/training/param_space.py` — RandomizedSearchCV distributions per model.
- `ml/training/experiment.py` — `run_experiment(disease)`: stratified split → 5-fold
  StratifiedKFold comparison → tuning → documented selection → refit. Writes
  `reports/<disease>/model_comparison.csv` + `SELECTION.md` (scaffolding only in Phase 2).
- `ml/evaluation/{metrics,curves,calibration}.py` — full metric suite, curve plots,
  Brier + calibration comparison.
- Decide diabetes train/test split strategy given the 25,772 duplicates.
- Unit tests: preprocessor fits only on train; no leakage; pipelines are picklable.

Reproduce Phase 1 from scratch: see `README.md` → "Reproduce Phase 1".

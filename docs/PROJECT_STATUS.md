# Project Status — Multi-Disease AI

_Last updated: 2026-09-02 (Phase 4)_

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
| 2 | Shared leakage-safe preprocessing + training/evaluation package | ✅ complete (`3b0ffde`) |
| 3 | **Heart** model end-to-end (CV, tuning, calibration, SHAP, model card) | ✅ trained |
| 4 | **Kidney** model end-to-end | ✅ trained |
| 5 | **Diabetes** model end-to-end | ⬜ next |
| 6 | External validation (Heart → Statlog) | ⬜ |
| 7 | Calibration + fairness analysis | ⬜ |
| 8 | FastAPI backend (`/predict/*`, `/models`, `/analytics/*`, `/scenario/*`) | ⬜ |
| — | React frontend, CNN/Grad-CAM imaging | deferred |

**Two models trained (heart, kidney). Every metric in this repo comes from an actual run — none are fabricated.**

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

## Phase 1 verification status (all from venv)

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

## Phase 2 results — shared ML package

Leakage-safe preprocessing + training/evaluation package in `ml/`, used by every disease
pipeline. Built without training anything; first exercised end-to-end in Phase 3.

| module | what it provides |
|---|---|
| `ml/preprocessing/build_preprocessor.py` | `build_preprocessor(spec)` → unfitted `ColumnTransformer`: numeric = median-impute + `StandardScaler`; categorical = most-frequent-impute + `OneHotEncoder(handle_unknown="ignore")`; binary = most-frequent-impute (unscaled). `verbose_feature_names_out=False`; `output_feature_names()` helper for SHAP/model cards. |
| `ml/training/splits.py` | `make_split(disease, X, y)` → single hold-out (`TEST_SIZE=0.20`, `RANDOM_STATE=42`). **Diabetes** groups identical feature rows (hash) and keeps each group on one side via `GroupShuffleSplit`; `cv_splitter("diabetes")` → `GroupKFold(shuffle=True)`. Heart/kidney/statlog → `train_test_split(stratify=y)` / `StratifiedKFold`. Indices returned sorted; `SplitResult.meta` records overall vs train vs test positive rate. |
| `ml/training/model_zoo.py` | `build_model(name, spec, smote=False)` → `imblearn.pipeline.Pipeline` `[preprocess, (smote), clf]`. Models: `logreg`, `random_forest`, `svc`, `xgboost`, `lightgbm`. SMOTE and `class_weight="balanced"` are mutually exclusive (SMOTE on ⇒ default weights). SMOTE only ever resamples inside a CV fold's train portion. |
| `ml/training/param_space.py` | `param_space(name, smote=)` — modest `RandomizedSearchCV` distributions (`clf__*`, `smote__k_neighbors`); `search_iter(name, disease)` — smaller `n_iter` for diabetes. |
| `ml/training/experiment.py` | `run_experiment(disease)` orchestration: load → `make_split` → `cross_validate_models` (5-fold) → `rank_models` (composite 0.4·ROC-AUC + 0.4·PR-AUC + 0.2·recall, **not accuracy**) → `tune_model` top-k → documented selection → refit. Writes `reports/<disease>/model_comparison.csv` + `SELECTION.md`. `selection_margin()` reports whether the winning gap exceeds fold-to-fold noise. |
| `ml/evaluation/metrics.py` | `classification_metrics()` (accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC, Brier, confusion counts), `threshold_sweep()`, `confusion_at()`, `youden_threshold()`. |
| `ml/evaluation/curves.py` | ROC / precision-recall / confusion-matrix / calibration-curve PNGs (`matplotlib` Agg), `all_evaluation_plots()`. |
| `ml/evaluation/calibration.py` | `compare_calibration()` — raw vs sigmoid (Platt) vs isotonic via `CalibratedClassifierCV` cross-fitted on **train only**, scored on test (Brier + ROC-AUC); `pick_calibration()` picks lowest Brier within `auc_tol` of raw. |

### Diabetes split-strategy decision

Naive random split rejected: 25,772 rows share a full 21-feature vector with another row,
so a model could score partly by memorising rows seen in both splits. **Decision:**
group identical feature vectors (`pd.util.hash_pandas_object` → `factorize`) and keep each
group wholly on one side of the split.

- Hold-out: `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`.
- CV: `GroupKFold(n_splits=5, shuffle=True, random_state=42)`.
- `StratifiedGroupKFold` was tried and **rejected on performance**: ~227,900 groups over
  253,680 rows (nearly every row unique) makes its greedy class-balancing ≈ O(n_groups²)
  — one split measured at ≈140 s, which made the test suite take 12+ min.
- Stratification is not enforced but is empirically fine at this group granularity:
  measured hold-out positive rates 0.1396 train / 0.1383 test vs 0.1393 overall;
  5-fold CV fold rates 0.138–0.141. `SplitResult.meta` records the real numbers each run.

Verified: 0 feature-vector hashes shared across the diabetes split. Heart/kidney/statlog
have 0 duplicate rows → plain stratified split.

### Phase 2 verification

| step | command | result |
|---|---|---|
| full suite (venv), as of Phase 2 | `python -m pytest -q` | **62 passed in 16.4 s** |

Runs in the venv (numpy 2.5.2, scikit-learn 1.7.2, imbalanced-learn 0.14.0). The 13
warnings are matplotlib/pyparsing deprecation noise pulled in via seaborn — cosmetic.

New tests: `tests/test_preprocessing.py` (fit_transform leaves no NaN / non-finite;
numeric imputer's `statistics_` equals the **train** median, not the full-data median;
`OneHotEncoder` tolerates unseen test categories; feature names exposed),
`tests/test_pipeline_leakage.py` (split disjoint + deterministic + ~stratified; no shared
diabetes feature vectors; correct CV splitter per disease; model pipelines are `imblearn`
Pipelines with a `ColumnTransformer` first step; SMOTE⇔class_weight exclusivity;
joblib round-trip), `tests/test_evaluation_metrics.py` (metric arithmetic on hand-built
arrays — no model involved).

## Phase 3 results — Heart model (trained)

**The first trained model.** Every number below is from `python -m scripts.train_heart`
and is reproducible with `RANDOM_STATE = 42`. Artifacts: `models/heart/`,
`reports/heart/`, `notebooks/heart_training.ipynb`.

New modules: `ml/explainability/shap_explainer.py`, `ml/model_card.py`,
`ml/training/finalize.py`, `ml/reporting.py`, `scripts/train_heart.py`.

### Split and selection

- 303 rows → 242 train / 61 test, stratified. Positive rate 0.4587 train / 0.4590 test.
- Model comparison (5-fold CV on train), by composite rank score: logreg 0.8789,
  random_forest 0.8713, xgboost 0.8630, svc 0.8528, lightgbm 0.8345.
- Top 3 tuned (RandomizedSearchCV, ROC-AUC): **xgboost 0.9079**, logreg 0.9074,
  random_forest 0.9061. Highest tuned score selected → `xgboost`.
- **The selection margin is not meaningful and is reported as such.** xgboost beat
  logreg by 0.0005 CV ROC-AUC while the fold-to-fold std is 0.0316 — 60× larger.
  `selection_margin.gap_within_cv_noise = true`; `SELECTION.md` and the model card both
  carry a prominent warning that these models perform comparably and a different seed
  could reorder them.

### Calibration (chosen on training out-of-fold scores only)

| method | train OOF Brier | test Brier |
|---|---|---|
| raw | 0.129782 | 0.1003 |
| **sigmoid (chosen)** | **0.128458** | 0.0893 |
| isotonic | 0.130883 | 0.0808 |

Worth noting: isotonic has the best *test* Brier. It was **not** chosen, because the
selection uses training out-of-fold scores where sigmoid won. Selecting on the test
column would have been test-set peeking and would have made the reported test metrics
optimistic. The test column is published for transparency only.

### Held-out test performance (n = 61)

| metric | threshold 0.5 | threshold 0.6436 |
|---|---|---|
| ROC-AUC | 0.9535 | 0.9535 |
| PR-AUC | 0.9496 | 0.9496 |
| recall / sensitivity | 0.8929 | 0.8214 |
| specificity | 0.8788 | 0.9394 |
| precision | 0.8621 | 0.9200 |
| F1 | 0.8772 | 0.8679 |
| accuracy | 0.8852 | 0.8852 |
| Brier | 0.0893 | 0.0893 |

The second threshold maximises Youden's J on **out-of-fold training** predictions and is
then applied unchanged to the test set — it is not tuned on the test set.

**Read these with the sample size in mind:** 61 test patients. A single reclassified
patient moves accuracy by 1.6 points. The cohort is referral-based (patients already
sent for angiography), so the 46% positive rate is far above any general population and
the probabilities are calibrated to that cohort, not the public.

### Explainability

TreeExplainer on the uncalibrated pipeline; SHAP values summed from the one-hot columns
back onto original feature names. **Additivity max error 1.4e-6** — the explanation
reconstructs the model's own output rather than approximating it. Top features by
mean |SHAP|: `thal` 0.569, `ca` 0.552, `cp` 0.509, `sex` 0.194, `slope` 0.193.

Two pipelines are saved: `pipeline.joblib` (calibrated, used for probabilities) and
`base_pipeline.joblib` (uncalibrated, used for SHAP — `CalibratedClassifierCV` hides the
tree structure SHAP needs). Platt scaling is monotonic, so contribution signs and
ranking carry over; the caveat is in the model card.

### Phase 3 verification

| step | command | result |
|---|---|---|
| training | `python -m scripts.train_heart` | exit 0, all artifacts written |
| full suite | `python -m pytest -q` | **74 passed, 24 skipped in 11 s** |
| notebook | `jupyter nbconvert --execute notebooks/heart_training.ipynb` | executes end-to-end |

The 24 skips are `tests/test_model_artifacts.py` for kidney and diabetes, which have no
trained model yet — they activate automatically in Phases 4-5.

`tests/test_model_artifacts.py` reloads the serialised pipeline, re-scores the test
split and asserts the result equals `metrics.json` — a fabricated or stale metric cannot
pass. It also checks SHAP additivity, that the alternative threshold was not chosen on
the test set, and that the model card carries a disclaimer and non-empty limitations.

## Phase 4 results — Kidney model (trained)

`python -m scripts.train_kidney`. Artifacts: `models/kidney/`, `reports/kidney/`,
`notebooks/kidney_training.ipynb`.

### Held-out test performance (n = 80)

| metric | threshold 0.5 | threshold 0.5792 |
|---|---|---|
| ROC-AUC | 0.9987 | 0.9987 |
| PR-AUC | 0.9992 | 0.9992 |
| recall / sensitivity | 0.9600 | 0.9600 |
| specificity | 0.9667 | 0.9667 |
| precision | 0.9796 | 0.9796 |
| accuracy | 0.9625 | 0.9625 |
| Brier | 0.0127 | 0.0127 |

Model `svc`, calibration `raw`. SHAP (KernelExplainer) additivity error 1.1e-16.
Top features: `sg` 0.076, `pcv` 0.070, `hemo` 0.066, `al` 0.048 — the standard CKD
laboratory markers.

**These numbers are close to meaningless as evidence of clinical usefulness.** Three
independent reasons are measured and recorded in the model card:

### 1. Missingness is confounded with the outcome

New in this phase: `ml/evaluation/diagnostics.py` runs a **missingness-only probe** —
discard every measured value, train logistic regression on 24 binary
*is-this-value-missing* indicators alone:

| | ROC-AUC |
|---|---|
| missingness indicators only, 5-fold CV | 0.8600 ± 0.0445 |
| missingness indicators only, test | 0.8023 |
| full model, test | 0.9987 |

So "which labs were ordered" alone reaches 0.80. Supporting evidence in
`reports/kidney/missingness_vs_target.csv`: `rbc` missing → 94.1% CKD vs 43.2% when
present; `rbcc` 94.7% vs 46.8%; `wbcc` 93.4% vs 51.4%. Complete-case analysis is not a
fix — only 158/400 rows (39.5%) are complete, and the positive rate collapses from
0.625 to 0.272 in that subset, so dropping incomplete rows would discard most of the
positive class.

The real model does add a lot (0.9987 vs 0.8023), so the labs carry genuine signal. But
a fifth of the separability is available from the measurement pattern alone, and because
imputation runs inside the pipeline, an imputed value *is* a marker that the test was
never ordered. The split stays clean — this is not train/test leakage — but it caps
transfer to any setting where these labs are ordered routinely.

### 2. Every model saturates the metric

CV ROC-AUC across the whole zoo: random_forest 0.9996, lightgbm 0.9990, svc 1.0000,
xgboost 0.9971, logreg 0.9990. The choice of algorithm is arbitrary and is reported as
such.

### 3. The labels are the diagnosis

The target is a recorded clinical diagnosis and the features include the criteria used
to make it. The model re-derives the diagnostic rule; it has no prognostic value.

### Bug found and fixed: saturated metrics read as "decisive"

Kidney exposed a real defect in the Phase 3 selection-margin check. It only compared the
top-2 gap against the fold-to-fold standard deviation. On this dataset SVC scores exactly
1.0 with a standard deviation of **0.0**, so a 0.0002 gap satisfied `gap > std` and the
run reported the choice as *decisive* — the least trustworthy case possible, since a zero
standard deviation at saturation means the metric has no resolution left, not that the
estimate is precise.

`selection_margin` now checks three independent failure modes and reports every reason
to doubt: gap below fold-to-fold std, gap below a 0.005 resolution floor, and both
candidates above 0.99 (saturation). Both models are now correctly reported as
**not decisive**:

- heart: gap 0.0005 vs std 0.0316 → noise, and below the resolution floor
- kidney: gap 0.0002 → below the resolution floor, and both models saturated

Heart was retrained after the fix (identical metrics — the change affects only the
verdict wording). `tests/test_selection_and_diagnostics.py` carries a regression test
for the saturation case.

### Phase 4 verification

| step | command | result |
|---|---|---|
| training | `python -m scripts.train_kidney` | exit 0, all artifacts written |
| retrain heart after margin fix | `python -m scripts.train_heart` | exit 0, metrics unchanged |
| full suite | `python -m pytest -q` | **99 passed, 12 skipped in 24 s** |
| notebooks | `jupyter nbconvert --execute` ×2 | both execute end-to-end |

The 12 skips are `tests/test_model_artifacts.py` for diabetes, which activates in Phase 5.

### Note for Phase 8

The kidney model is an `SVC`, so SHAP falls back to `KernelExplainer`, which is
**slow per call**. Generating a local explanation for a single request is not free the
way it is for the tree-based heart model. Either precompute, cache, or accept the latency
— decide when building `/predict/kidney`.

## Next step — Phase 5 (Diabetes model end-to-end)

`scripts/train_diabetes.py` calling `train_disease("diabetes")`. Different challenges from
the first two: 253,680 rows (CV cost — reduce `n_iter`, prefer LightGBM/XGBoost), 14%
positive (report PR-AUC prominently, compare SMOTE against class weights), group-aware
split already in place, and SHAP on a background sample for tractability.

Reproduce Phase 1 from scratch: see `README.md` → "Reproduce Phase 1".

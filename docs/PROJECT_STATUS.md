# Project Status — Multi-Disease AI

_Last updated: 2026-09-04 (Phase 11 — accounts and saved history)_

> **Looking for the current state rather than the build history?** See
> [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — condensed orientation: the three models,
> their real metrics, what drives each score, known limitations, and the next step.
> This file is the phase-by-phase log.

Research/education platform. **Not** a clinical diagnostic tool. Three independent
disease pipelines (heart, kidney, diabetes), a FastAPI backend serving them, and a React
frontend over that. Authentication is mandatory: nothing but the landing and account
screens is reachable without a session. Image/OCR extraction and the CNN/Grad-CAM module
remain deferred.

Plan file: `C:\Users\Admin\.claude\plans\you-are-working-on-zippy-hammock.md`

## Phase progress

| Phase | Scope | State |
|---|---|---|
| 0 | Repo scaffold, environment, `config.py` | ✅ complete (`f64caa9`, `e3a06b4`) |
| 1 | Dataset acquisition, caching, loaders, EDA, target definitions | ✅ complete (`f64caa9`, `e3a06b4`) |
| 2 | Shared leakage-safe preprocessing + training/evaluation package | ✅ complete (`3b0ffde`) |
| 3 | **Heart** model end-to-end (CV, tuning, calibration, SHAP, model card) | ✅ trained |
| 4 | **Kidney** model end-to-end | ✅ trained |
| 5 | **Diabetes** model end-to-end | ✅ trained |
| 6 | External validation (Heart → Statlog) | ✅ attempted, REJECTED |
| 7 | Calibration + fairness subgroup analysis | ✅ complete |
| 8 | FastAPI backend (`/predict/*`, `/models`, `/analytics/*`, `/scenario/*`) | ✅ complete (`739eeac`) |
| 9 (v1) | Frontend: guided form, review step, worked examples | ✅ complete |
| 10 | Upload-first flow: PDF extraction, synthetic reports, report PDF, redesign | ✅ complete (`805bbfd`) |
| 11a | Auth backend: argon2id, opaque server-side sessions, reset, rate limits | ✅ complete (`ea2de55`) |
| 11b | Lock the API down; retrofit the existing suite to a signed-in client | ✅ complete (`4558efe`) |
| 11c | Landing → sign in → `/app`; every route re-parented behind one gate | ✅ complete (`b3cd96e`) |
| 11d | Saved analyses and history, pinned to the model that produced them | ✅ complete (`82eb39e`) |
| 11e | Hardening, `docs/AUTH.md`, README/API/FRONTEND updates | ✅ complete |
| — | Paste report text; image/OCR extraction | ⬜ |
| — | CNN/Grad-CAM imaging module | deferred |

**All three models trained (heart, kidney, diabetes). Every metric in this repo comes from an actual run — none are fabricated.**

**Authentication is mandatory; storage is not.** Signing in unlocks the assessment flow. It
does not start a record: `/predict/*` writes nothing, an uploaded document is never stored,
and health values reach the database only through the explicit save on the result screen.
`tests/test_access_control.py` runs a full assessment and asserts every table's row count is
unchanged. See [`AUTH.md`](AUTH.md).

## Environment

- Python 3.12.6. Project runs in its own venv: `.venv/` (git-ignored).
- venv key versions: numpy 2.5.2, pandas 2.3.0, scikit-learn 1.7.2, xgboost 3.0.5,
  lightgbm 4.6.0, imbalanced-learn 0.14.0, shap 0.52.0, numba 0.67.0, ucimlrepo 0.0.7,
  matplotlib 3.10.3, seaborn 0.13.2, fastapi 0.115.0, uvicorn 0.30.6, pydantic 2.9.1,
  sqlalchemy 2.0.44, alembic 1.17.1, argon2-cffi 25.1.0, email-validator 2.3.0.
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
| statlog (**rejected** as external validation — Cleveland subset, see Phase 6) | 145 | 270 | 13 | `target == 2` | 150 / 120 | 0.4444 | 1.25 |

Target transformations and feature dictionaries: `reports/<disease>/TARGET.md`.

### Missing values

- **heart**: `ca` 4 (1.3%), `thal` 2 (0.7%). 6 rows affected.
- **kidney**: every column has some; 1,012 cells; 242/400 rows (60.5%) affected.
  Heaviest — `rbc` 38.0%, `rbcc` 32.8%, `wbcc` 26.5%, `pot` 22.0%, `sod` 21.8%, `pcv` 17.8%.
- **diabetes**: none.
- **statlog**: none.

No imputation at load time — deferred to the in-pipeline preprocessing (Phase 2) to avoid leakage.

### Other EDA findings

- **diabetes has 25,772 duplicate rows** in the `df.duplicated()` sense — i.e. rows beyond
  the first in each repeated profile. Counted the other way, **38,000 rows (15.0%) share an
  identical 21-feature vector with at least one other row**, spread over 12,228 groups.
  (Both numbers are correct; they answer different questions. Phase 5 uses the second.)
  Legitimate for a survey, not dropped. Flag for Phase 2: a naive random split can place
  identical feature vectors in both train and test — decide on split strategy.
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
4. ~~Statlog feature encodings (cp/slope/thal) are *assumed* compatible with Cleveland for
   Phase 6 external validation — to be verified when that phase runs.~~
   **Resolved in Phase 6.** The encodings *are* compatible (verified: identical value
   domains, marginals within ~1pp). But the check also revealed something worse — Statlog
   is a 270-row **subset of Cleveland** (100% exact feature+label match, 82.2% inside the
   heart training split), so it cannot serve as external validation at all. See the
   Phase 6 section below.

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

Naive random split rejected: 38,000 rows share a full 21-feature vector with at least one
other row, so a model could score partly by memorising rows seen in both splits. **Decision:**
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
| specificity | 0.9667 | **1.0000** |
| precision | 0.9796 | **1.0000** |
| F1 | 0.9697 | **0.9796** |
| accuracy | 0.9625 | **0.9750** |
| Brier | 0.0127 | 0.0127 |

_(Corrected in Phase 5: this table previously repeated the threshold-0.5 column under the
0.5792 heading. The values above are from `reports/kidney/metrics.json`. ROC-AUC, PR-AUC and
Brier are threshold-free and genuinely identical across both columns; the four count-based
metrics are not. Raising the threshold to 0.5792 removes the 1 false positive, leaving the
2 false negatives untouched.)_

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

## Phase 5 results — Diabetes model (trained)

`python -m scripts.train_diabetes`. Artifacts: `models/diabetes/`, `reports/diabetes/`,
`notebooks/diabetes_training.ipynb`. New modules: `ml/training/imbalance.py`; new
diagnostics `duplicate_label_conflict_probe` and `operating_point_note`.

The first two diseases were small (n=303, n=400) and easy. This one is large (253,680),
imbalanced (13.9% positive) and genuinely hard — the interesting work was in reporting a
moderate result honestly rather than dressing it up.

### Held-out test performance (n = 50,961)

| metric | threshold 0.5 | threshold 0.1388 |
|---|---|---|
| ROC-AUC | 0.8317 | 0.8317 |
| PR-AUC | 0.4307 | 0.4307 |
| recall / sensitivity | 0.1464 | **0.7984** |
| specificity | 0.9829 | 0.7090 |
| precision | 0.5791 | 0.3056 |
| F1 | 0.2338 | 0.4421 |
| accuracy | 0.8673 | 0.7213 |
| Brier | 0.0959 | 0.0959 |

Model `xgboost` (766 trees, depth 4, lr 0.027), calibration `isotonic`, TreeExplainer
additivity error 3.3e-6. Top features by mean |SHAP|: `GenHlth` 0.653, `HighBP` 0.444,
`BMI` 0.415, `Age` 0.404, `HighChol` 0.298 — the known correlates, in a sensible order.

Selection is again **not decisive**: xgboost beat lightgbm by 0.0003 CV ROC-AUC against a
fold-to-fold std of 0.0014, and below the 0.005 resolution floor.

### 0.5 is the wrong threshold here, and the card says so

Recall of **0.1464** at threshold 0.5 looks like a broken model. It is not. Isotonic
calibration maps scores onto true probabilities, and at 13.8% prevalence a calibrated
probability only exceeds 0.5 for extreme profiles — so 0.5 behaves as a high-precision
screen. The uncalibrated CV recall was 0.795 and the training-derived threshold 0.1388
(≈ prevalence) recovers 0.798, which is consistent.

New `operating_point_note` fires when the two thresholds differ by more than 0.20 recall
and states which row to read. It stays quiet for heart, where 0.5 is a fine default.

### SMOTE was measured and it lost

`ml/training/imbalance.py` cross-validates the selected model under both strategies on
identical folds (SMOTE inside the pipeline, resampling each fold's training portion only):

| strategy | ROC-AUC | PR-AUC | recall@0.5 | precision@0.5 | fit s |
|---|---|---|---|---|---|
| **class_weight (shipped)** | **0.8301** | **0.4354** | 0.7951 | 0.3065 | 32.0 |
| smote | 0.8239 | 0.4243 | 0.2899 | 0.5012 | 65.9 |

PR-AUC −0.0111 against a fold std of 0.0082, at 2.1× the fit cost. The common advice does
not hold here, and there is now a measurement rather than an assumption in either direction.

### SVC excluded on measured cost, not quietly dropped

Fit times on subsamples: 1.07 s @ n=2,000 → 4.27 s @ 4,000 → 17.52 s @ 8,000 → 85.09 s @
16,000. Empirical exponent ≈2.28, extrapolating to **~7.7 h for one fit** on 202,719
training rows and **~23 h for a single 5-fold CV pass**, before tuning. `EXCLUDED_MODELS`
in `model_zoo.py` records the reason, which is reproduced in `SELECTION.md` and the model
card: excluded on cost, never scored, nothing claimed about how it would have done.

Random-forest's search space is also narrowed for diabetes only (`n_estimators` 150–400
instead of 200–800) — a cost axis, not a modelling choice; documented in `param_space.py`.

### The ceiling imposed by the feature set

21 mostly-binary survey answers cannot distinguish 253,680 people. **38,000 rows (15.0%)**
share an identical feature vector with at least one other row; 1,566 of those groups
contain both labels, making 1,640 rows unwinnable for any model on these features —
a hard accuracy ceiling of **0.9935**.

Measured accuracy is 0.8673, i.e. **0.1262 below the bound**, so the ceiling does *not*
explain the shortfall. `duplicate_conflict_note` deliberately refuses to absolve a model
that far below the bound; a test pins that behaviour. Irreducible error is only an
explanation when the model is near it, otherwise it is an excuse.

### Calibration (chosen on training out-of-fold Brier only)

| method | train OOF Brier | test Brier |
|---|---|---|
| raw | 0.174663 | 0.1736 |
| sigmoid | 0.096932 | 0.0962 |
| **isotonic (chosen)** | **0.096620** | 0.0959 |

Here train and test agree, so the guard is not load-bearing — unlike heart, where it
stopped isotonic being chosen on its test score.

### SHAP at scale

Global importance is averaged over a **random 5,000-row sample of the 50,961 test rows**
(`max_explain_rows`, `random_state=42`), recorded in `shap_global.json` as `n_rows_explained`
plus an `explained_on` string so it cannot be mistaken for the full test set. No-op for
heart (61) and kidney (80).

### Phase 5 verification

| step | command | result |
|---|---|---|
| training | `python -m scripts.train_diabetes` | exit 0, all artifacts written |
| reproducibility | retrained all 3 after adding the note | heart/kidney/diabetes metrics **identical** |
| full suite | `python -m pytest -q` | **129 passed, 0 skipped in 74 s** |
| notebooks | `jupyter nbconvert --execute` ×3 training | 18/18 cells each, 0 errors |

First run with **zero skips** — all three diseases now have artifacts, so
`tests/test_model_artifacts.py` is fully active. 18 new tests in
`tests/test_scale_and_imbalance.py`.

## Notes for Phase 8 (backend)

1. **The kidney model is an `SVC`** → SHAP falls back to `KernelExplainer`, slow per call.
   Precompute, cache, or accept the latency on `/predict/kidney`.
2. **`RISK_BANDS` are prevalence-agnostic and misleading for diabetes.** Measured band
   distribution on each test set:

   | disease | low | moderate | high |
   |---|---|---|---|
   | heart | 44.3% | 14.8% | 41.0% |
   | kidney | 36.2% | 3.8% | 60.0% |
   | diabetes | **86.8%** | 12.6% | **0.6%** |

   A correctly calibrated 13.9%-prevalence model almost never exceeds 0.66, so "high risk"
   is nearly unreachable and "low" is nearly universal. Fixed 0.33/0.66 cut-points encode
   an implicit assumption of balanced prevalence. Options for Phase 8: per-disease bands
   derived from the training distribution (e.g. probability quantiles), bands relative to
   prevalence, or dropping bands for diabetes in favour of a percentile. **Not changed in
   Phase 5** — it is a presentation decision affecting all three modules and the frontend.
3. Both `pipeline.joblib` (calibrated, for probabilities) and `base_pipeline.joblib`
   (uncalibrated, for SHAP) must be loaded per disease at startup.

## Phase 6 results — External validation ATTEMPTED AND REJECTED

`python -m ml.external.heart_statlog`. Artifacts: `reports/heart/external_validation.json`,
`EXTERNAL_VALIDATION.md`, `external_schema_check.csv`, `notebooks/external_validation.ipynb`.
New modules: `ml/external/dataset_overlap.py`, `ml/external/heart_statlog.py`.

Two checks ran before any metric was believed.

**Check 1 — encoding compatibility: passed.** The Phase 1 assumption holds. `cp` 1–4,
`restecg` 0–2, `slope` 1–3, `ca` 0–3, `thal` 3/6/7 in both releases; marginals agree within
~1pp; numeric ranges match exactly.

**Check 2 — dataset independence: failed decisively.**

| check | result |
|---|---|
| Statlog rows matching a Cleveland row on all 13 features | **270 / 270 (100%)** |
| of those, labels also identical | **270 / 270** |
| rows in the heart model's **training** split | **222 (82.2%)** |
| rows in the heart test split | 48 (17.8%) |
| rows unseen by the model | **0** |

Matching is 1:1 (neither dataset has a duplicate feature vector). Statlog is a redistributed
subset of Cleveland, not a second cohort.

Scored naively it reports **ROC-AUC 0.9399** vs the internal test's 0.9535 — *slightly below*,
which is exactly what a sound external validation looks like. That is the danger: the
contaminated result does not look broken. It is kept in `EXTERNAL_VALIDATION.md`, labelled
and not to be cited.

**The heart model has no external validation.** Recorded in its model card (both as a
dedicated section and in the limitations list) and carried through retraining via
`finalize._existing_external_validation`.

**A bug worth recording:** the first overlap check reported **0% overlap** — a false negative
in the most dangerous direction, caused by Cleveland loading as `int64` and Statlog as
`float64` so string keys compared `"63"` to `"63.0"`. Fixed in `row_keys()` (cast to float
first) and pinned by `test_row_keys_match_across_int_and_float_dtypes`.

### Phase 6 verification

| step | command | result |
|---|---|---|
| external validation | `python -m ml.external.heart_statlog` | exit 0, status REJECTED |
| card durability | `python -m scripts.train_heart` | verdict survives retrain; metrics unchanged |
| full suite | `python -m pytest -q` | **145 passed, 0 skipped in 66 s** |
| notebook | `jupyter nbconvert --execute notebooks/external_validation.ipynb` | 7/7 cells, 0 errors |

## Phase 7 results — Calibration + subgroup performance

`python -m scripts.fairness_report`. New modules: `ml/fairness/subgroup_metrics.py`,
`scripts/fairness_report.py`, plus `expected_calibration_error` /
`plot_reliability_comparison` / `calibration_summary` in `ml/evaluation/calibration.py`.
Artifacts per disease: `reports/<disease>/fairness/{subgroups.csv,disparities.json,FAIRNESS.md}`,
`reports/<disease>/calibration/summary.json`, `figures/calibration_before_after.png`,
plus `notebooks/calibration_fairness.ipynb`.

### Calibration, before → after (test set)

| disease | wrapper | Brier | ECE | Δ ROC-AUC |
|---|---|---|---|---|
| heart | sigmoid | 0.1003 → 0.0893 (−0.0110) | 0.1236 → 0.0765 (−0.0471) | +0.0032 |
| kidney | raw | 0.0127 → 0.0127 (0.0000) | 0.0362 → 0.0362 (0.0000) | 0.0000 |
| diabetes | isotonic | 0.1736 → 0.0959 (−0.0777) | 0.2356 → **0.0020** (−0.2336) | −0.0002 |

ECE isolates calibration where Brier mixes it with discrimination; reported next to Brier,
never instead (ECE is blind to ranking). The diabetes row explains the Phase 5 threshold
finding: `scale_pos_weight` left the raw model badly overconfident at ECE 0.2356, and
isotonic cut that ~117×, which is why calibrated probabilities stop crossing 0.5 at 13.8%
prevalence. Kidney's zero change is correct — `raw` was the selected method.

### Subgroups — heart and kidney support no claims

| disease | test n | attributes | reliable subgroups |
|---|---|---|---|
| heart | 61 | sex, age_band | **1 / 6** |
| kidney | 80 | age_band only (no sex variable recorded) | **0 / 4** |
| diabetes | 50,961 | sex, age_band | 6 / 6 |

Reliable = ≥20 rows, ≥10 positives, ≥10 negatives. Rates carry Wilson 95% intervals,
ROC-AUC a Hanley–McNeil interval. Three of heart's age bands report recall exactly
**1.0000** on intervals like [0.4385, 1.0000] — three or four patients each. Every heart
and kidney comparison returns inconclusive, and a regression test asserts none is ever
reported otherwise. The symmetric trap is stated in each write-up: too little data to show
a disparity is not evidence there isn't one.

### Two bars for calling a disparity

Disjoint intervals **and** gap ≥ `PRACTICAL_GAP = 0.05`. The second bar is the large-sample
mirror of the small-sample problem — at n=50,961 almost any difference separates. The
diabetes sex ROC-AUC gap (0.8419 F vs 0.8179 M, 0.0240) is statistically clear and
correctly reported as small, not material.

### Diabetes findings that hold

| age band | n | ROC-AUC | recall @0.5 | recall @0.1388 | specificity @0.1388 |
|---|---|---|---|---|---|
| 18-34 | 4,848 | **0.8583** | 0.0084 | 0.2689 | 0.9852 |
| 35-49 | 10,111 | 0.8535 | 0.0791 | 0.6109 | 0.8806 |
| 50-64 | 18,110 | 0.8241 | 0.1669 | 0.7820 | 0.7099 |
| 65+ | 17,892 | **0.7683** | 0.1481 | 0.8603 | 0.5019 |

1. Discrimination declines monotonically with age (0.8583 → 0.7683, gap 0.0900, disjoint).
   Threshold-free and identical at both cut-offs — the robust finding.
2. A single global threshold gives completely different error profiles by age: at 0.1388
   the 65+ band gets recall 0.860 / specificity 0.502 against 18-34's 0.269 / 0.985.
3. The recall gap swings 0.1585 → 0.5914 between the two thresholds, so subgroup analysis
   is run at **both** — reporting one would have misrepresented it.

Caveat recorded: age is one of the model's strongest features, so within-band AUC removes
most of that variance and is not comparable to the overall 0.8317.

**Nothing here labels a model fair.** Only sex and age are available, the labels carry
their own bias, and parity on two attributes says nothing about who a model harms.

### Phase 7 verification

| step | command | result |
|---|---|---|
| fairness + calibration | `python -m scripts.fairness_report` | exit 0, all artifacts written |
| full suite | `python -m pytest -q` | **170 passed, 0 skipped in 73 s** |
| notebook | `jupyter nbconvert --execute notebooks/calibration_fairness.ipynb` | 7/7 cells, 0 errors |

## Phase 8 — FastAPI backend — COMPLETE

Three independent modules behind one API. No combined endpoint, no aggregate health score,
and **no training, fitting or dataset read in any request path**. Full reference:
[`docs/API.md`](API.md).

```
backend/
├── main.py                        app, CORS, lifespan model loading, OpenAPI description
├── schemas/features.py            request models GENERATED from serving.json
├── schemas/responses.py           response models
├── services/registry.py           all artifacts loaded once at startup
├── services/prediction_service.py request -> row -> probability -> explanation -> caveats
└── routes/{health,models,predict,analytics,scenario,deps}.py
```

Startup measured at **~7 s** for three pipelines, three model cards and three SHAP
explainers, dominated by the kidney `KernelExplainer`. Building explainers eagerly is
deliberate: lazily would move that cost onto whichever request arrived first.

### The API reads `models/`, never `data/`

`python -m scripts.export_serving_assets` writes two files per disease after training:

| file | contents |
|---|---|
| `serving.json` | operating threshold, risk bands, per-feature ranges/categories, explainer profile, external-validation status |
| `shap_background.joblib` | the exact 200 training rows the training run's explainer used |

Ranges are taken from the **training** split only, so nothing the API advertises about its
own inputs was derived from held-out data. Exporting the same background rows (same
`RANDOM_STATE`, same sample size) means the served explanation is the one that was
validated, not a re-derived approximation.

### The risk-band defect, found and fixed

The fixed 0.33/0.66 bands in `config.py` were wrong for diabetes, and the size of the error
is worth recording. On the 50,961-row diabetes test set:

| band | rows | share |
|---|---|---|
| low | 44,243 | 86.8% |
| moderate | 6,435 | 12.6% |
| high | 283 | 0.56% |

The model's operating threshold is 0.1388, which sits deep inside the "low" band — so
**every case the model flags, including every true positive it catches, was labelled "low"
or "moderate"**. The presentation label contradicted the model's own decision.

Bands are now anchored on each model's threshold `t`: `[0, t/2)`, `[t/2, t)`, `[t, 1]`.
`flagged == (risk_band == "high")` holds by construction and is pinned by tests in both
`tests/test_api.py` and `tests/test_model_artifacts.py`. Only the upper boundary carries
meaning; the response's `note` field says the lower one is arbitrary.

`config.risk_band(probability, threshold)` now takes the threshold with **no default** — a
band computed without one is meaningless, so it is not possible to ask for one.

Model cards were amended in place (`_amend_risk_bands`) rather than retrained: only that
block changed, and `_card_markdown` is a pure function of the card dict.

### Findings from Phases 3-7 carried into the response body

| finding | where it surfaces |
|---|---|
| risk bands broken for diabetes | bands anchored on the operating threshold; `flagged` reported separately from the label |
| 0.5 is the wrong decision point | `threshold` + `threshold_rule` on every response; 0.5 is never used |
| kidney SHAP is slow (809 ms, measured) | published in `/models`; `?include_explanation=false`; off by default on `/scenario` |
| no module has external validation | a warning on every prediction; heart's says *rejected*, not *absent* |
| error profiles differ by age band | the case's own band's measured recall/specificity, where Phase 7 marked it reliable |
| every response needs the disclaimer | `disclaimer` on every response and in the OpenAPI description |
| SHAP explains the uncalibrated model | `uncalibrated_probability` returned next to `probability`, with a note |

Heart and kidney contribute nothing to the subgroup line: 1 of 6 and 0 of 4 of their
subgroups were reliable on 61 and 80 test rows. An unreliable subgroup produces silence.

### Input validation: three levels, not two

Rejected (422): outside the mechanical envelope, unrecognised categorical level, unknown
field, or *no fields at all*. Accepted and flagged (200): a value inside the envelope but
outside the observed training range, and any omitted field.

Unknown categories are **rejected rather than warned about** because the one-hot encoder
runs with `handle_unknown="ignore"` — an unrecognised level is silently encoded as
all-zeros and would return a confident-looking prediction for a case whose category the
model never saw. That is the failure mode worth spending an error code on.

An empty body is refused: every field is optional, so it would otherwise be scored entirely
from imputed training medians and returned as a prediction of nothing. A *partially*
specified case is a legitimate research question and is answered, with the imputation
listed.

The envelope (one observed range either side of the training min/max) is mechanical and
encodes **no clinical knowledge**. `GET /models/{disease}/schema` returns both it and the
observed training range, and `range_note` says which is which.

### Request schemas are generated, not typed

`backend/schemas/features.py` builds the three pydantic models from each `serving.json` at
import. Fifty-eight fields with individual bounds and category lists is exactly the kind of
thing that accumulates silent transcription errors, and a bound that disagrees with the
training data is worse than no bound — it would reject valid inputs or admit ones the
encoder cannot represent. Generating them means the API can only advertise ranges that were
measured.

### Phase 8 verification

| step | command | result |
|---|---|---|
| serving assets | `python -m scripts.export_serving_assets` | 3 diseases, thresholds 0.6436 / 0.5792 / 0.1388 |
| app boots | `uvicorn backend.main:app --port 8123` | `/health` `ok`, `/docs` 200, `/openapi.json` 200 |
| live prediction | `POST /predict/heart` (Cleveland row 0) | p=0.3666, flagged=false, band=moderate, SHAP returned |
| full suite | `python -m pytest -q` | **224 passed, 0 skipped in 61 s** |

170 → 224: 48 new API tests plus 6 new risk-band tests.

## Phase 9 (v1) — Frontend: guided form + worked examples — COMPLETE

Vite + React 18 + TypeScript + Tailwind, no component or charting library. 191 KB bundle
(61 KB gzipped). Full write-up: [`docs/FRONTEND.md`](FRONTEND.md).

```bash
uvicorn backend.main:app --port 8000     # terminal 1
npm --prefix frontend run dev            # terminal 2  ->  http://localhost:5173
```

### Why a form before the document upload

The requested headline was drag-and-drop upload with OCR. It could not be v1, because **no
document-based entry mode can complete any of the three models**: kidney gets ~20 of 24 from
a perfect urinalysis + CBC, heart gets 5 of 13 and none of its top three (`thal`, `ca`, `cp`
need a nuclear scan, a catheter angiogram and a clinician's history), and diabetes gets ~5 of
21 because it is a survey model whose strongest feature is self-rated health. Every path ends
at a form for the remainder, and the review screen *is* that form pre-filled — so paste-text
and OCR upload are pre-fills for what this phase built, not alternatives to it.

### Making 58 fields feel like 8

| lever | effect |
|---|---|
| pick one module first | kidney never shows a diabetes survey question |
| `tier` from the model's own SHAP | core = smallest set covering 80% of mean \|SHAP\| — **7 / 8 / 7** fields, not 13 / 24 / 21 |
| groups by source document | *From your urine test*, *From an exercise stress test*, … — how the paper is actually held |
| coverage weighted by attribution mass | kidney `sg` (16.4%) moves the bar 80x further than `su` (0.2%) |

The form is **generated** from `GET /models/{disease}/schema` — no input, label, option list
or bound is hand-written, so a retrain cannot leave the UI describing a model that no longer
exists.

### Safety carried into the UI

* **"I don't have this" is a first-class answer**, distinct from an empty box. Both reach the
  model as missing, but only one is a decision — and the review step refuses to run while
  anything is merely untouched. This matters most for kidney, where missingness is confounded
  with the outcome (a missingness-indicators-only model reaches ROC-AUC 0.8023).
* **The decision is the headline, not the probability.** A bare probability invites a mental
  50% cut-off; for diabetes that turns recall 0.7984 into 0.1464.
* Imputed fields are named, extrapolated values flagged, every API warning shown, and the
  model's full limitations list rendered on the result page.
* Session state is in memory only — nothing entered is persisted, and any edit invalidates a
  prediction made from the previous values.

### New serving assets

`export_serving_assets` now also writes per-feature `label`, `group`, `tier`, `shap_rank` and
`shap_mass_share`, plus `models/<disease>/samples.json` — real rows from the **held-out test
split** with their recorded outcome, two per class. `ml/serving/field_groups.py` holds the
hand-authored presentation metadata (groups, labels, coded value labels), kept separate from
everything measured, with `LABEL_SOURCES` recording provenance — including an explicit note
that only three of the eight BRFSS Income levels are pinned by the cached UCI metadata and
the rest follow the codebook unverified.

### Three defects found by running it

1. **Ordinal survey codes rendered as number boxes.** `Age`, `GenHlth`, `Education`, `Income`
   are modelled as numeric, so they had no options — the form asked people to type `8` for
   their age band. Now labelled dropdowns that still send the number; `options_for` raises if
   any level in the observed range is unlabelled.
2. **Scroll position carried across routes**, landing users below the decision headline.
3. **The auto-opened group could be one filtered out** by "key fields only" — for kidney,
   every visible section rendered collapsed.

### Phase 9 (v1) verification

| step | command | result |
|---|---|---|
| typecheck | `npx tsc --noEmit` | clean |
| build | `npm run build` | 190.78 KB JS / 61.36 KB gzip, 0 errors |
| kidney end-to-end | browser, sample case | flagged, p=1.0000 at t=0.5792, 21/24 fields, SHAP rendered |
| diabetes end-to-end | browser, sample case | **p=14.5% vs t=13.9% -> flagged, band High** — the case the old fixed bands would have called "low risk" |
| full suite | `python -m pytest -q` | **260 passed, 0 skipped in 62 s** |

224 -> 260: 36 new tests over the intake metadata and sample cases.

## Phase 10 — Upload-first assessment flow — COMPLETE

Backend spine committed as `6625553`; the frontend redesign sits on top of it. Full write-up:
[`docs/FRONTEND.md`](FRONTEND.md).

The requested journey — **choose → upload → review → analyse → explain → report** — with the
guided form kept as the fallback. Upload, paste and manual entry all write the same session
state and produce a byte-identical backend request; the review screen is the single
verified-data surface. Phase 8 API contracts are untouched; everything added is additive.

### Extraction, and the test that makes it real

Deterministic pattern matching against each model's own schema. No LLM: for a text-layer PDF
with a label/value layout it is more reliable and, unlike a language model, cannot invent a
value. Three invariants, each pinned:

* an unfound label or unparseable value returns `missing` with `value=None` — never a median
  or a plausible number;
* a coded field can only take a level the model was trained on;
* an out-of-range number is surfaced as `needs_review`, not clipped.

The central test hands the extractor **only the PDF bytes** and asserts every extracted value
equals the row the document was rendered from: **0 mismatches across all 12 reports**. Without
it, a working-looking demo could simply be echoing data it already had.

The blood-pressure split is explicit per field: `heart.trestbps` takes the systolic half of a
"142/90" pair, `kidney.bp` the diastolic. Training medians are 130 and 80; feeding one into
the other's model would be silently, confidently wrong.

### Synthetic demonstration reports

Values from a real **held-out test row** the model was never fitted on; the document around
them fabricated and marked as such on its face. So extraction runs on a genuine document and
the prediction at the end is a genuine prediction on an unseen record. Kidney reports
naturally carry unrecorded labs (3, 1, 0, 0 across the four), which is what produces the
"complete what's missing" step rather than a staged one.

### Two deliberate departures from the brief

**The category is the headline, not the percentage.** The category's upper boundary *is* the
operating threshold, so it cannot contradict the model's decision — a bare percentage read
against an assumed 50% can. For diabetes that is recall 0.7984 vs 0.1464.

**The dropzone advertises PDF only.** The brief listed "PDF, JPG or PNG", but image
extraction is not implemented and the server answers a JPG with 415. An affordance that does
not work is worse than one that is absent.

### Density

Removed from the primary path: coverage meter, "key field" badges, feature-count
percentages, landing-page metrics, raw SHAP values, long paragraphs. **Almost none deleted** —
moved to `/:disease/about` and the result page's disclosures. Only `CoverageMeter.tsx` and the
coverage helpers went for good.

### Phase 10 verification

| step | command | result |
|---|---|---|
| demo reports | `python -m ml.reports.demo_report` | 12 PDFs, 1 page each, text layer extracts |
| extraction round-trip | `pytest tests/test_extraction.py` | 24 passed, **0 mismatches / 12 reports** |
| typecheck + build | `npx tsc --noEmit`, `npm run build` | clean, 201 KB (64 KB gzip) |
| kidney end to end | browser, real dropzone | 21/24 read, gate held on 3, **High · 100.0% · above 57.9%**, report 200 |
| heart end to end | browser, real dropzone | 13/13 read, **High · 69.0% · above 64.4%**, report 200 |
| diabetes end to end | browser, real dropzone | 21/21 read, **High · 14.5% · above 13.9%**, report 200 |
| full suite | `python -m pytest -q` | **311 passed, 0 skipped in 63 s** |

260 -> 311: 28 extraction tests, 23 document-endpoint tests.

Two further extraction defects surfaced by cross-uploading a report to the wrong module, and
both were root-caused rather than patched at the symptom. "Cholesterol checked in the last 5
years - Yes" prefix-matches the `chol` alias and the parser harvested the `5` out of "5
years"; a value must now follow its label directly. And "Age group 75-79" yielded a heart age
of 75 marked *found*; a bare range where a single value is expected is now refused. Neither
affected the demo path -- all 12 reports still round-trip with 0 mismatches -- but both would
have produced a confident wrong number on a real document.

Uploading to the wrong module is now safe field by field: only `age` (years) and `sex` (same
0/1 convention in both models that have it) carry across, since those are the same
measurement wherever they appear. Diabetes's 1-13 `Age` band deliberately does not, so an age
in years cannot land in it.

## Next step — remaining work

* **Paste report text** — the same extractor minus the PDF step.
* **Image/OCR extraction** — needs Tesseract, and the PHI questions answered if any cloud
  service is involved.
* **CNN/Grad-CAM imaging module** — still deferred, "Coming Soon" until a real model exists.

## Next step — deferred work

Neither is started, and neither is required for the platform to be complete as scoped.

* **React frontend** (Vite/Tailwind/Recharts) consuming these endpoints: landing, three
  independent disease cards, per-disease forms, result screen (prediction → SHAP → scenario
  → model card), analytics pages. It must show `flagged` and `threshold`, not a bare
  probability against 0.5, and must not aggregate the three modules into one score.
* **CNN + Grad-CAM imaging module**, architecturally separate. "Coming Soon" until a real
  model exists — no placeholder predictions.

### Superseded — the original Phase 6 plan, kept for the record

> `ml/external/heart_statlog.py`: align Statlog's 13 features to Cleveland names/encodings,
> map target 1/2 → 0/1, load `models/heart/pipeline.joblib` and predict **with no
> retraining**. Write `reports/heart/external_validation.json` plus a write-up comparing
> internal-test against external performance. The encoding compatibility of `cp`/`slope`/
> `thal` is currently an *assumption* (see Known issues #4) and must be verified as part of
> that phase — if it does not hold, the external result is meaningless and should be
> reported as such rather than published as a validation.

The guard in that last sentence is what fired, though not for the reason anticipated. The
encodings were fine; the dataset was not independent. Worth noting that the plan's
verification step ("`python -m ml.external.heart_statlog` → `external_validation.json`")
would have been satisfied by a fully contaminated result — the file gets written either
way. Independence had to be checked explicitly; it was not implied by any step of the plan.

Reproduce Phase 1 from scratch: see `README.md` → "Reproduce Phase 1".

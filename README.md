# HealthLens

**Explainable machine-learning risk prediction for heart disease, chronic kidney disease
and diabetes, built on public datasets.** Upload a health report or fill in a short form,
and see what stands out, in everyday language. The project and its code are named
*Multi-Disease AI*; HealthLens is the web app.

> **For research and educational purposes only.** Predictions come from models trained on
> historical public datasets and must **not** be read as a medical diagnosis or medical
> advice.

## Screenshots

<table>
  <tr>
    <td width="50%"><img src="docs/screenshots/landing-light.png" alt="HealthLens landing page in light mode, with the rotating DNA helix"></td>
    <td width="50%"><img src="docs/screenshots/landing-dark.png" alt="HealthLens landing page in dark mode"></td>
  </tr>
  <tr>
    <td align="center"><sub>Landing page, light</sub></td>
    <td align="center"><sub>Landing page, dark</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/screenshots/sign-in.png" alt="Sign-in screen"></td>
    <td width="50%"><img src="docs/screenshots/sign-up.png" alt="Create-an-account screen in dark mode"></td>
  </tr>
  <tr>
    <td align="center"><sub>Sign in</sub></td>
    <td align="center"><sub>Create an account</sub></td>
  </tr>
</table>

<p align="center">
  <img src="docs/screenshots/landing-mobile.png" alt="HealthLens landing page on a phone" width="260"><br>
  <sub>On a phone</sub>
</p>

## Modules

Three **independent** pipelines, each with its own dataset, preprocessing, model
comparison, evaluation and explanations. Their predictions are **never** combined into one
"overall health score".

| Module | Dataset | Target |
|---|---|---|
| Heart Disease Presence | UCI Heart Disease (Cleveland), id 45 | angiographic disease present (`num > 0`) |
| Chronic Kidney Disease Presence | UCI Chronic Kidney Disease, id 336 | `class == 'ckd'` |
| Diabetes Health-Indicator Risk | UCI CDC Diabetes Health Indicators (BRFSS 2015), id 891 | reported prediabetes or diabetes |

## Good to know

- **Sign-in is required.** Every assessment, upload and report needs an account. Health
  values are stored only when you choose to save them. See [docs/AUTH.md](docs/AUTH.md).
- **Every number is real.** All metrics come from actual training runs (`RANDOM_STATE = 42`);
  none are placeholders. Each model's uses and limits are in `models/<disease>/MODEL_CARD.md`.
- **No external validation.** Heart's intended external cohort (Statlog, UCI 145) turned out
  to be a copy of rows already in Cleveland, so it was rejected. All performance figures come
  from one held-out split of one dataset.
- **Explanations come with every result.** SHAP shows which values pushed the estimate higher
  or lower, mapped back to the original field names.

## Repository layout

```
📦 medicl
├── 🧠 ml/                    the machine-learning package
│   ├── data/                 dataset download, caching, loaders, validation
│   ├── eda/                  profiling and plots
│   ├── preprocessing/        leakage-safe pipeline per disease
│   ├── training/             model zoo, search spaces, experiment runners
│   ├── evaluation/           metrics, curves, calibration
│   ├── explainability/       SHAP, global and per-prediction
│   ├── fairness/             subgroup analysis
│   ├── external/             the Heart → Statlog validation attempt
│   ├── extraction/           PDF report → form fields (never invents a value)
│   ├── reports/              sample reports and the downloadable result PDF
│   └── serving/              labels, units and field groups the app shows
│
├── 🔌 backend/               FastAPI service: routes, schemas, accounts, sessions
├── 💻 frontend/              React + Vite + Tailwind web app (HealthLens)
│
├── 📊 models/<disease>/      model cards, metadata and serving files
├── 📈 reports/<disease>/     EDA, metrics, figures
├── 📓 notebooks/             walkthrough notebooks (read results, never retrain)
├── 🧾 demo_reports/          synthetic sample PDFs to try the upload flow
├── 🔧 scripts/               training and export entry points
├── 📜 migrations/            database schema (Alembic)
├── ✅ tests/                 structure, leakage and shipped-model tests
│
├── 📚 docs/                  API.md · FRONTEND.md · AUTH.md · screenshots/
└── config.py                 seed, paths, disease registry, disclaimer
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   ·   macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m alembic upgrade head          # creates the accounts database
```

Copy `.env.example` to `.env` only if you want to change a setting; every value has a
development default.

## Run the web app

```bash
uvicorn backend.main:app --reload       # backend on port 8000
npm --prefix frontend install
npm --prefix frontend run dev           # app on http://localhost:5173
```

Pick an assessment, upload a report (or enter values by hand), review what was found, then
see the result and what drove it. Sample reports are on the first screen. More in
[docs/FRONTEND.md](docs/FRONTEND.md).

## Methodology

- **No data leakage.** Data is split before anything is fitted; all imputation, scaling and
  resampling happen inside the pipeline and cross-validation folds.
- **No choices made on the test set.** Model, hyper-parameters, calibration and decision
  threshold are all picked on training folds.
- **Close results are reported as close.** None of the three models won decisively (margins of
  0.0005, 0.0002 and 0.0003 cross-validated ROC-AUC), and every model card says so.
- **The API can't invent a number.** It serves the saved training results, and no model is
  trained while handling a request.
- **Outliers are flagged, not dropped**, because medical measurements can be genuinely extreme.
- **The three diseases stay separate**: their own data, preprocessing, models and model cards.

"""FastAPI application for the Multi-Disease AI platform.

Three independent disease modules served behind one API. There is no combined endpoint and
no aggregate health score: the models were trained on unrelated cohorts with unrelated
features, so any number combining them would mean nothing.

Design constraints, in order of importance:

* **No training in the request path, ever.** Pipelines, model cards and SHAP explainers are
  loaded once during the lifespan startup. A request does ``predict_proba`` and
  ``shap_values``, and nothing else.
* **No number without provenance.** Analytics endpoints read the artifacts written by the
  training runs; the API never recomputes or estimates a metric.
* **The caveats travel with the prediction.** The disclaimer, the model's
  external-validation status, what was imputed, what was extrapolated and (where Phase 7
  measured it reliably) the subgroup error profile are all part of the response body, not
  documentation somewhere else.

Run locally::

    uvicorn backend.main:app --reload
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import analytics, documents, health, models, predict, scenario
from backend.services.registry import load_registry
from config import DISCLAIMER

DESCRIPTION = f"""
**{DISCLAIMER}**

Three independent risk-prediction modules, each with its own dataset, preprocessing, model,
evaluation and explainability layer:

| module | dataset | positive class |
|---|---|---|
| `heart` | UCI Heart Disease, Cleveland (n=303) | angiographic heart disease present |
| `kidney` | UCI Chronic Kidney Disease (n=400) | chronic kidney disease present |
| `diabetes` | CDC BRFSS Diabetes Health Indicators (n=253,680) | prediabetes **or** diabetes reported |

They are never combined. There is no overall health score.

### Reading a prediction

* `flagged` is the model's decision, taken at **its own operating threshold** — not 0.5.
  For diabetes that threshold is 0.1388, and at 0.5 the model's recall on the test set is
  0.1464 rather than 0.7984. A caller that thresholds the probability at 0.5 itself will
  get a very different, and much worse, classifier.
* `risk_band` is a presentation label anchored on the same threshold. It is not a clinical
  category and not the decision.
* `explanation` comes from SHAP on the **uncalibrated** base model; the note in the payload
  says exactly how that relates to the calibrated `probability`.
* **No module has external validation.** Heart's intended external cohort (Statlog) was
  tested and rejected as a redistributed subset of its own training data. Every performance
  figure comes from one held-out split of one dataset.

### What this is not

Not a diagnostic system, not a screening tool, and not a forecast of future disease. The
diabetes module in particular is trained on self-reported survey answers and a self-reported
diagnosis, so it learns who has *been told* they have diabetes — which tracks access to
healthcare as well as disease.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every model artifact once, before the first request."""
    app.state.registry = load_registry()
    status = app.state.registry.status()
    for disease, s in status.items():
        if s.get("loaded"):
            print(
                f"[startup] {disease}: {s['model']} ({s['calibration']}), "
                f"threshold {s['operating_threshold']:.4f}, "
                f"explainer {s['explainer']} "
                f"{'ok' if s['explainer_available'] else 'FAILED: ' + str(s['explainer_error'])} "
                f"({s['load_seconds']}s)"
            )
        else:
            print(f"[startup] {disease}: NOT LOADED — {s['error']}")
    yield
    app.state.registry = None


app = FastAPI(
    title="Multi-Disease AI — Explainable Risk Prediction",
    description=DESCRIPTION,
    version="0.8.0",
    lifespan=lifespan,
)

# Local development only. `ALLOWED_ORIGINS` (comma-separated) must be set to the real
# frontend origin before this is exposed anywhere; a wildcard is not a deployment setting.
_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(predict.router)
app.include_router(analytics.router)
app.include_router(scenario.router)
app.include_router(documents.router)


@app.get("/", tags=["health"], summary="Service description and disclaimer")
def root():
    return {
        "service": "Multi-Disease AI — Explainable Risk Prediction",
        "version": app.version,
        "modules": list(getattr(app.state, "registry", None).diseases)
        if getattr(app.state, "registry", None)
        else [],
        "docs": "/docs",
        "disclaimer": DISCLAIMER,
    }

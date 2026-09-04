"""FastAPI application behind HealthLens (project: Multi-Disease AI).

Three independent disease modules served behind one API. There is no combined endpoint and
no aggregate health score: the models were trained on unrelated cohorts with unrelated
features, so any number combining them would mean nothing.

Design constraints, in order of importance:

* **No training in the request path, ever.** Pipelines, model cards and SHAP explainers are
  loaded once during the lifespan startup. A request does ``predict_proba`` and
  ``shap_values``, and nothing else.
* **No number without provenance.** Analytics endpoints read the artifacts written by the
  training runs; the API never recomputes or estimates a metric.
* **No health information without an account.** Every route except ``/health``, ``/``
  and ``/auth/*`` requires a session cookie. There is no anonymous prediction path.
* **Authentication does not imply storage.** Signing in unlocks the assessment flow; it
  does not cause anything to be written. Health values reach the database only through an
  explicit save, and ``/predict/*`` never writes at all.
* **The caveats travel with the prediction.** The disclaimer, the model's
  external-validation status, what was imputed, what was extrapolated and (where Phase 7
  measured it reliably) the subgroup error profile are all part of the response body, not
  documentation somewhere else.

Run locally::

    uvicorn backend.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect

from backend.db import engine, session_factory
from backend.routes import (
    analyses, analytics, auth, documents, health, models, predict, scenario,
)
from backend.routes.deps import require_user
from backend.services import auth_service
from backend.services.registry import load_registry
from backend.settings import settings, validate
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

### Authentication

Every endpoint below except `/health`, `/` and `/auth/*` requires a session. Sign in through
`POST /auth/login`; the session is an opaque token in an httpOnly cookie, sent automatically
by the browser. There is no bearer token and no anonymous access to any module.

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
    """Validate configuration, check the schema, then load every model artifact once."""
    problems = validate(settings)
    if problems:
        listed = "".join(f"\n  - {p}" for p in problems)
        if settings.is_production:
            raise RuntimeError(f"Refusing to start with this configuration:{listed}")
        print(f"[startup] configuration warnings:{listed}")

    # A missing schema otherwise surfaces as an opaque OperationalError on the first signup.
    if not inspect(engine()).has_table("users"):
        print(
            "[startup] WARNING: the accounts schema is missing. "
            "Run `alembic upgrade head` — sign-in will fail until you do."
        )
    else:
        with session_factory()() as db:
            swept = auth_service.sweep_expired(db)
        if any(swept.values()):
            print(f"[startup] swept expired rows: {swept}")

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
    title="HealthLens — Explainable Risk Prediction",
    description=DESCRIPTION,
    version="0.11.0",
    lifespan=lifespan,
    # The interactive docs are a development convenience. They are the one thing here that
    # invites a stranger to start poking at the auth routes, so production turns them off.
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
)

# Credentials are now required, so the origin list must be exact: browsers reject a
# wildcard origin on a credentialed request, and `validate()` refuses to start production
# with one. In development the Vite proxy makes `/api` same-origin, so this path is not
# exercised at all — which is why it is worth getting right rather than discovering it at
# deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)

# Authentication is enforced here, once, rather than on individual endpoints. Attaching
# `require_user` to whole routers means a route added to any of them later is protected by
# default instead of by being remembered — and `tests/test_access_control.py` sweeps every
# registered route to prove it, so a new router added to the wrong tuple fails the suite
# rather than shipping open.
app.include_router(health.router)
app.include_router(auth.router)

for _router in (models.router, predict.router, analytics.router, scenario.router,
                documents.router, analyses.router):
    app.include_router(_router, dependencies=[Depends(require_user)])


@app.get("/", tags=["health"], summary="Service description and disclaimer")
def root():
    return {
        "service": "HealthLens — Explainable Risk Prediction",
        "version": app.version,
        "modules": list(getattr(app.state, "registry", None).diseases)
        if getattr(app.state, "registry", None)
        else [],
        "docs": "/docs",
        "disclaimer": DISCLAIMER,
    }

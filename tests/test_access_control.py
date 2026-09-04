"""Phase 11b tests: nothing but the account routes is reachable without a session.

The important test here is :func:`test_every_route_outside_the_public_set_requires_a_session`,
which sweeps the routes the application actually registered rather than a list written by
hand. Authentication attached to a router protects the endpoints added to it later, but only
if the router itself was included in the protected tuple — this is what makes forgetting
that a test failure instead of an open endpoint.

The second half pins the promise that goes with mandatory sign-in: **being authenticated
does not mean being recorded.** Running an assessment writes nothing. Health values reach
the database only through the explicit save flow.
"""
from __future__ import annotations

import pytest
from fastapi.routing import APIRoute
from sqlalchemy import func, select

from backend.main import app
from backend.schemas.features import example_payload
from backend.tables import LoginAttempt, PasswordReset, User, UserSession

# Reachable without a session, and deliberately short. `/health` is a liveness probe, `/`
# is the service description and disclaimer, `/auth/*` is how you get a session at all, and
# the docs are development-only (settings.docs_enabled turns them off in production).
PUBLIC_PATHS = {"/", "/health", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
PUBLIC_PREFIXES = ("/auth/",)

# Concrete values for path parameters, so the sweep hits a real route rather than a 404.
PATH_VALUES = {"disease": "heart", "case_id": "heart-positive-1"}

TABLES = (User, UserSession, PasswordReset, LoginAttempt)


def _is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def _concrete(path: str) -> str:
    for name, value in PATH_VALUES.items():
        path = path.replace("{" + name + "}", value)
    # Anything still templated gets a placeholder; it only needs to route, not to succeed.
    while "{" in path:
        head, _, rest = path.partition("{")
        _, _, tail = rest.partition("}")
        path = head + "placeholder" + tail
    return path


def _protected_routes() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or _is_public(route.path):
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            out.append((method, route.path))
    return sorted(set(out))


def test_the_sweep_actually_found_the_protected_routes():
    """Guard against the sweep silently covering nothing."""
    routes = _protected_routes()
    assert len(routes) >= 12, routes
    paths = {path for _, path in routes}
    for expected in ("/predict/heart", "/extract/{disease}", "/models", "/analytics/{disease}"):
        assert expected in paths, f"{expected} missing from the sweep"


@pytest.mark.parametrize("method,path", _protected_routes())
def test_every_route_outside_the_public_set_requires_a_session(anon_client, method, path):
    response = anon_client.request(method, _concrete(path), json={})
    assert response.status_code == 401, (
        f"{method} {path} answered {response.status_code} without a session"
    )
    assert "Sign in" in response.json()["detail"]


def test_the_public_set_is_exactly_what_it_should_be():
    """A new public route must be a deliberate edit to this list, not a side effect."""
    public = {
        route.path
        for route in app.routes
        if isinstance(route, APIRoute) and _is_public(route.path)
    }
    assert public == {
        "/",
        "/health",
        "/auth/signup",
        "/auth/login",
        "/auth/logout",
        "/auth/me",
        "/auth/sessions",
        "/auth/logout-all",
        "/auth/password",
        "/auth/forgot",
        "/auth/reset",
        "/auth/account",
    }


def test_health_and_root_stay_reachable_without_a_session(anon_client):
    assert anon_client.get("/health").status_code == 200
    assert anon_client.get("/").status_code == 200


def test_an_invalid_session_cookie_is_refused_like_no_cookie_at_all(anon_client):
    anon_client.cookies.set("mda_session", "not-a-real-token")
    try:
        r = anon_client.post("/predict/heart", json=example_payload("heart"))
        assert r.status_code == 401
    finally:
        anon_client.cookies.clear()


def test_a_signed_in_client_can_still_do_everything_it_could_before(client):
    assert client.get("/models").status_code == 200
    assert client.post("/predict/heart", json=example_payload("heart")).status_code == 200
    assert client.get("/analytics/heart").status_code == 200


# --- authentication is not surveillance ---------------------------------------------------


def _row_counts(db) -> dict[str, int]:
    return {
        table.__tablename__: int(db.execute(select(func.count()).select_from(table)).scalar_one())
        for table in TABLES
    }


def test_running_an_assessment_writes_nothing_to_the_database(client, db):
    """Signing in unlocks the flow. It does not start a record.

    Predict, explain and scenario are pure functions of their input. If this test ever
    fails, the API has begun retaining health information as a side effect of being used,
    which is a different product from the one described to the user.
    """
    before = _row_counts(db)

    for disease in ("heart", "kidney", "diabetes"):
        payload = example_payload(disease)
        assert client.post(f"/predict/{disease}", json=payload).status_code == 200
    assert client.post(
        "/scenario/heart",
        json={"features": example_payload("heart"), "overrides": {"age": 71}},
    ).status_code == 200

    db.expire_all()
    assert _row_counts(db) == before


def test_uploading_a_document_writes_nothing_to_the_database(client, db):
    from config import ROOT_DIR

    pdf = ROOT_DIR / "demo_reports" / "heart-positive-1.pdf"
    if not pdf.exists():
        pytest.skip("demo reports not generated")

    before = _row_counts(db)
    response = client.post(
        "/extract/heart",
        files={"file": ("heart-positive-1.pdf", pdf.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["n_found"] > 0

    db.expire_all()
    assert _row_counts(db) == before

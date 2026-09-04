"""Shared test fixtures.

``DATABASE_URL`` is set **at import time**, before anything under ``backend`` is imported,
because :mod:`backend.settings` reads the environment once at module import. pytest loads
``conftest.py`` before the test modules, so this is the only place the redirect can happen
reliably — and it must happen, or a test run would write accounts into the development
database in ``data/app.db``.

Each run gets a fresh SQLite file in a temporary directory, created with
:func:`backend.db.create_all` rather than by running Alembic: the migration is verified
separately (``test_auth.py::test_migration_and_orm_agree_on_the_schema``), and rebuilding
the schema per run keeps a broken migration from making every unrelated test fail.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from itertools import count
from pathlib import Path

_TMP_DIR = Path(tempfile.mkdtemp(prefix="medicl-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_DIR / 'test.db').as_posix()}"
os.environ.setdefault("ENV", "development")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:5173")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.db import create_all, engine, session_factory  # noqa: E402
from backend.main import app  # noqa: E402

create_all()

_counter = count(1)

PASSWORD = "correct-horse-battery"


@pytest.fixture(scope="session", autouse=True)
def _cleanup_database():
    yield
    engine().dispose()
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


@pytest.fixture
def db():
    session = session_factory()()
    try:
        yield session
    finally:
        session.close()


def unique_email(prefix: str = "user") -> str:
    """A fresh address per call, so one test's rate-limit ledger cannot affect another's."""
    # Not example.com/.test/.invalid: those are reserved special-use names and
    # email-validator refuses them, which would fail on the address rather than on
    # anything this suite is testing.
    return f"{prefix}-{next(_counter)}@medicl-test.org"


@pytest.fixture(scope="module")
def client():
    """Unauthenticated client. The registry loads once per module (~7 s with explainers)."""
    with TestClient(app) as c:
        yield c


def register(c: TestClient, *, email: str | None = None, password: str = PASSWORD,
             display_name: str | None = None) -> dict:
    """Create an account on this client. The session cookie is kept by the client's jar."""
    body = {"email": email or unique_email(), "password": password}
    if display_name:
        body["display_name"] = display_name
    response = c.post("/auth/signup", json=body)
    assert response.status_code == 201, response.text
    return {"email": body["email"], "password": password, **response.json()}


@pytest.fixture(scope="module")
def auth_client():
    """A client that is signed in for the whole module.

    Most of the suite is testing model behaviour, not authentication, and every one of
    those routes now requires a session. This fixture is what they use.
    """
    with TestClient(app) as c:
        register(c)
        yield c

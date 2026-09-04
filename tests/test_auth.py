"""Phase 11a tests: accounts, sessions and the properties the auth routes must hold.

Ordinary contract testing is the smaller half of this file. The rest pins the properties
that are invisible when the feature "works" and expensive when they are wrong:

* a password is never stored, returned or logged, and neither is a usable session token;
* the two sign-in failure modes are indistinguishable in status, message **and** time;
* changing or resetting a password ends the other sessions;
* deleting an account really deletes it, including by cascade — which on SQLite depends on
  a per-connection PRAGMA that is silently absent if forgotten;
* the Alembic migration and the ORM describe the same schema.

The client fixture deliberately does **not** enter ``TestClient`` as a context manager. That
would run the application lifespan and load three pipelines plus a KernelExplainer (~7 s)
for tests that never touch a model.
"""
from __future__ import annotations

import os
import time
from datetime import timedelta

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, inspect, select

from backend.db import Base, session_factory
from backend.main import app
from backend.settings import settings
from backend.tables import LoginAttempt, PasswordReset, User, UserSession, utcnow
from config import ROOT_DIR
from tests.conftest import PASSWORD, register, unique_email

COOKIE = settings.cookie_name


@pytest.fixture
def c() -> TestClient:
    """A fresh client — and therefore a fresh cookie jar — with no model registry loaded."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_rate_limit_ledger():
    """Independent tests.

    The per-IP login limit counts failures from ``testclient``, which every test in the
    suite shares. Without this, a test that deliberately fails a sign-in five times would
    push unrelated tests over the limit — and the failure would land somewhere else.
    """
    with session_factory()() as db:
        db.execute(delete(LoginAttempt))
        db.commit()
    yield


def _cookie_header(response) -> str:
    return response.headers.get("set-cookie", "")


# --- signing up and in -------------------------------------------------------------------


def test_signup_creates_an_account_and_signs_in(c):
    email = unique_email()
    r = c.post("/auth/signup", json={"email": email, "password": PASSWORD,
                                     "display_name": "Varun"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["user"]["email"] == email
    assert body["user"]["greeting_name"] == "Varun"
    assert c.cookies.get(COOKIE)

    me = c.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == email


def test_greeting_name_falls_back_to_the_local_part_not_the_address(c):
    account = register(c, email="quiet.person@medicl-test.org")
    assert account["user"]["greeting_name"] == "quiet.person"
    assert "@" not in account["user"]["greeting_name"]


def test_me_is_204_when_not_signed_in(c):
    r = c.get("/auth/me")
    assert r.status_code == 204
    assert not r.content


def test_email_case_does_not_create_a_second_account(c):
    email = unique_email()
    register(c, email=email.upper())
    r = c.post("/auth/signup", json={"email": email, "password": PASSWORD})
    assert r.status_code == 409


def test_surrounding_whitespace_is_stripped_before_an_address_is_stored():
    from backend.services.auth_service import normalize_email
    assert normalize_email("  Person@Example.ORG  ") == "person@example.org"


def test_signing_in_is_case_insensitive(c):
    email = unique_email()
    register(c, email=email)
    c.post("/auth/logout")
    r = c.post("/auth/login", json={"email": email.upper(), "password": PASSWORD})
    assert r.status_code == 200


def test_logout_clears_the_cookie_and_deletes_the_session(c, db):
    register(c)
    raw = c.cookies.get(COOKIE)
    assert db.execute(select(UserSession)).scalars().all()

    r = c.post("/auth/logout")
    assert r.status_code == 204
    assert 'mda_session=""' in _cookie_header(r) or "mda_session=;" in _cookie_header(r)
    assert c.get("/auth/me").status_code == 204

    from backend.security import token_digest
    assert db.execute(
        select(UserSession).where(UserSession.token_hash == token_digest(raw))
    ).scalar_one_or_none() is None


def test_a_revoked_session_cookie_stops_working_immediately(c, db):
    register(c)
    raw = c.cookies.get(COOKIE)
    db.execute(delete(UserSession))
    db.commit()

    fresh = TestClient(app)
    fresh.cookies.set(COOKIE, raw)
    assert fresh.get("/auth/me").status_code == 204


# --- what must never leave the server ----------------------------------------------------


def test_no_response_ever_contains_the_password_or_its_hash(c, db):
    email = unique_email()
    signup = c.post("/auth/signup", json={"email": email, "password": PASSWORD})
    user = db.execute(select(User).where(User.email == email)).scalar_one()

    for response in (signup, c.get("/auth/me"), c.get("/auth/sessions")):
        text = response.text
        assert PASSWORD not in text
        assert user.password_hash not in text
        assert "password_hash" not in text


def test_the_stored_password_is_an_argon2id_hash_not_the_password(c, db):
    email = unique_email()
    register(c, email=email)
    user = db.execute(select(User).where(User.email == email)).scalar_one()
    assert user.password_hash.startswith("$argon2id$")
    assert PASSWORD not in user.password_hash


def test_the_session_token_is_stored_only_as_a_digest(c, db):
    email = unique_email()
    register(c, email=email)
    raw = c.cookies.get(COOKIE)
    user = db.execute(select(User).where(User.email == email)).scalar_one()
    session = db.execute(
        select(UserSession).where(UserSession.user_id == user.id)
    ).scalar_one()
    assert session.token_hash != raw
    assert len(session.token_hash) == 64
    from backend.security import token_digest
    assert session.token_hash == token_digest(raw)


def test_the_session_cookie_is_httponly_and_samesite_lax(c):
    r = c.post("/auth/signup", json={"email": unique_email(), "password": PASSWORD})
    header = _cookie_header(r).lower()
    assert "httponly" in header
    assert "samesite=lax" in header
    assert "path=/" in header


def test_the_raw_token_is_only_ever_in_the_set_cookie_header(c):
    r = c.post("/auth/signup", json={"email": unique_email(), "password": PASSWORD})
    raw = c.cookies.get(COOKIE)
    assert raw and raw not in r.text


# --- failure modes are uniform ------------------------------------------------------------


def test_wrong_password_and_unknown_account_are_indistinguishable(c):
    email = unique_email()
    register(c, email=email)

    wrong = c.post("/auth/login", json={"email": email, "password": "not-the-password"})
    unknown = c.post("/auth/login", json={"email": unique_email(), "password": PASSWORD})

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"] == "Incorrect email or password."


def test_an_unknown_account_still_costs_a_password_verification(c):
    """Timing must not be an account-existence oracle.

    Without the dummy-hash verification in ``verify_password`` this path returns without
    hashing anything and answers in microseconds, while a real account costs an argon2
    verification. The floor is set far below argon2's real cost (tens of milliseconds) so
    the test measures the presence of the work, not the speed of the machine.
    """
    elapsed = []
    for _ in range(3):
        start = time.perf_counter()
        c.post("/auth/login", json={"email": unique_email(), "password": PASSWORD})
        elapsed.append(time.perf_counter() - start)
    assert min(elapsed) > 0.01, f"unknown-account sign-in returned too fast: {elapsed}"


def test_a_weak_or_reused_password_is_refused(c):
    for password in ("short", "password123", "x" * 200):
        r = c.post("/auth/signup", json={"email": unique_email(), "password": password})
        assert r.status_code == 422, password


def test_the_password_cannot_be_the_email_address(c):
    email = unique_email()
    r = c.post("/auth/signup", json={"email": email, "password": email})
    assert r.status_code == 422


# --- rate limiting -------------------------------------------------------------------------


def test_repeated_failures_are_locked_out_with_a_retry_after(c):
    email = unique_email()
    register(c, email=email)

    statuses = [
        c.post("/auth/login", json={"email": email, "password": "wrong-one-here"}).status_code
        for _ in range(6)
    ]
    assert statuses[:5] == [401] * 5
    assert statuses[5] == 429

    blocked = c.post("/auth/login", json={"email": email, "password": PASSWORD})
    assert blocked.status_code == 429, "the correct password must not bypass the lockout"
    assert int(blocked.headers["retry-after"]) > 0


def test_a_successful_sign_in_clears_earlier_failures(c):
    email = unique_email()
    register(c, email=email)
    for _ in range(4):
        c.post("/auth/login", json={"email": email, "password": "wrong-one-here"})
    assert c.post("/auth/login", json={"email": email, "password": PASSWORD}).status_code == 200
    # The four failures are gone, so five more are needed to lock out again.
    for _ in range(4):
        assert c.post("/auth/login", json={"email": email, "password": "wrong"}).status_code == 401


# --- changing a password -------------------------------------------------------------------


def test_changing_the_password_keeps_this_session_and_ends_the_others(c, db):
    email = unique_email()
    register(c, email=email)

    other = TestClient(app)
    assert other.post("/auth/login", json={"email": email, "password": PASSWORD}).status_code == 200
    assert other.get("/auth/me").status_code == 200

    r = c.post("/auth/password", json={"current_password": PASSWORD,
                                       "new_password": "a-brand-new-secret"})
    assert r.status_code == 200
    assert c.get("/auth/me").status_code == 200, "the browser making the change stays signed in"
    assert other.get("/auth/me").status_code == 204, "other browsers must be signed out"

    c.post("/auth/logout")
    assert c.post("/auth/login", json={"email": email, "password": PASSWORD}).status_code == 401
    assert c.post("/auth/login", json={"email": email,
                                       "password": "a-brand-new-secret"}).status_code == 200


def test_changing_the_password_requires_the_current_one(c):
    register(c)
    r = c.post("/auth/password", json={"current_password": "wrong-current",
                                       "new_password": "a-brand-new-secret"})
    assert r.status_code == 401


def test_logout_all_leaves_only_this_browser(c):
    email = unique_email()
    register(c, email=email)
    others = [TestClient(app) for _ in range(2)]
    for o in others:
        o.post("/auth/login", json={"email": email, "password": PASSWORD})

    r = c.post("/auth/logout-all")
    assert r.status_code == 200
    assert c.get("/auth/me").status_code == 200
    for o in others:
        assert o.get("/auth/me").status_code == 204
    assert len(c.get("/auth/sessions").json()) == 1
    assert c.get("/auth/sessions").json()[0]["current"] is True


# --- password reset -------------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.sent: list[tuple[str, str, str]] = []

    def send(self, to, subject, body):
        self.sent.append((to, subject, body))


@pytest.fixture
def mailer(monkeypatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr("backend.routes.auth.get_mailer", lambda: recorder)
    return recorder


def _link_token(recorder: _Recorder) -> str:
    body = recorder.sent[-1][2]
    line = next(l for l in body.splitlines() if "/reset/" in l)
    return line.rsplit("/reset/", 1)[1].strip()


def test_forgot_password_answers_the_same_whether_or_not_the_account_exists(c, mailer):
    email = unique_email()
    register(c, email=email)

    known = c.post("/auth/forgot", json={"email": email})
    unknown = c.post("/auth/forgot", json={"email": unique_email()})

    assert known.status_code == unknown.status_code == 202
    assert known.json() == unknown.json()
    assert len(mailer.sent) == 1, "no mail may be sent for an address with no account"


def test_a_reset_link_sets_a_new_password_and_ends_every_session(c, mailer):
    email = unique_email()
    register(c, email=email)
    assert c.get("/auth/me").status_code == 200

    c.post("/auth/forgot", json={"email": email})
    token = _link_token(mailer)

    r = c.post("/auth/reset", json={"token": token, "new_password": "recovered-password-1"})
    assert r.status_code == 200
    assert c.get("/auth/me").status_code == 204, "a reset must sign every browser out"

    fresh = TestClient(app)
    assert fresh.post("/auth/login", json={"email": email,
                                           "password": "recovered-password-1"}).status_code == 200


def test_a_reset_token_works_once(c, mailer):
    email = unique_email()
    register(c, email=email)
    c.post("/auth/forgot", json={"email": email})
    token = _link_token(mailer)

    assert c.post("/auth/reset", json={"token": token,
                                       "new_password": "recovered-password-1"}).status_code == 200
    second = c.post("/auth/reset", json={"token": token, "new_password": "another-password-2"})
    assert second.status_code == 400


def test_an_expired_reset_token_is_refused(c, db, mailer):
    email = unique_email()
    register(c, email=email)
    c.post("/auth/forgot", json={"email": email})
    token = _link_token(mailer)

    user = db.execute(select(User).where(User.email == email)).scalar_one()
    grant = db.execute(
        select(PasswordReset).where(PasswordReset.user_id == user.id)
    ).scalar_one()
    grant.expires_at = utcnow() - timedelta(minutes=1)
    db.commit()

    r = c.post("/auth/reset", json={"token": token, "new_password": "recovered-password-1"})
    assert r.status_code == 400


def test_requesting_a_new_link_invalidates_the_previous_one(c, mailer):
    email = unique_email()
    register(c, email=email)
    c.post("/auth/forgot", json={"email": email})
    first = _link_token(mailer)
    c.post("/auth/forgot", json={"email": email})
    second = _link_token(mailer)

    assert first != second
    assert c.post("/auth/reset", json={"token": first,
                                       "new_password": "recovered-password-1"}).status_code == 400
    assert c.post("/auth/reset", json={"token": second,
                                       "new_password": "recovered-password-1"}).status_code == 200


def test_a_reset_request_does_not_change_the_password_by_itself(c, mailer):
    email = unique_email()
    register(c, email=email)
    c.post("/auth/forgot", json={"email": email})
    fresh = TestClient(app)
    assert fresh.post("/auth/login", json={"email": email, "password": PASSWORD}).status_code == 200


# --- deleting an account ---------------------------------------------------------------------


def test_deleting_an_account_requires_the_password_and_the_confirmation(c):
    register(c)
    assert c.request("DELETE", "/auth/account",
                     json={"password": "wrong", "confirm": "DELETE"}).status_code == 401
    assert c.request("DELETE", "/auth/account",
                     json={"password": PASSWORD, "confirm": "delete"}).status_code == 422
    assert c.get("/auth/me").status_code == 200, "the account must still exist"


def test_deleting_an_account_removes_the_user_and_everything_owned_by_it(c, db, mailer):
    email = unique_email()
    register(c, email=email)
    c.post("/auth/forgot", json={"email": email})
    user = db.execute(select(User).where(User.email == email)).scalar_one()
    user_id = user.id
    assert db.execute(select(UserSession).where(UserSession.user_id == user_id)).scalars().all()
    assert db.execute(select(PasswordReset).where(PasswordReset.user_id == user_id)).scalars().all()

    r = c.request("DELETE", "/auth/account", json={"password": PASSWORD, "confirm": "DELETE"})
    assert r.status_code == 204
    assert c.get("/auth/me").status_code == 204

    db.expire_all()
    assert db.get(User, user_id) is None
    assert not db.execute(select(UserSession).where(UserSession.user_id == user_id)).scalars().all()
    assert not db.execute(select(PasswordReset).where(PasswordReset.user_id == user_id)).scalars().all()


def test_sqlite_actually_enforces_the_cascade(c, db):
    """Regression guard for a silent one.

    SQLite ignores ``ON DELETE CASCADE`` unless ``PRAGMA foreign_keys=ON`` is issued on the
    connection. Miss it and "delete my account" appears to succeed while leaving every
    session row — and, from Phase 11d, every saved analysis — behind.
    """
    email = unique_email()
    register(c, email=email)
    user = db.execute(select(User).where(User.email == email)).scalar_one()
    user_id = user.id

    db.execute(delete(User).where(User.id == user_id))  # raw delete, no ORM cascade
    db.commit()

    assert db.execute(
        select(UserSession).where(UserSession.user_id == user_id)
    ).scalars().all() == []


# --- schema ------------------------------------------------------------------------------------


def test_migration_and_orm_agree_on_the_schema(tmp_path):
    """``alembic upgrade head`` must produce exactly what the ORM describes.

    The suite builds its database with ``create_all``, so without this a column added to
    :mod:`backend.tables` and never migrated would pass every test here and fail on the
    first real deployment.
    """
    url = f"sqlite:///{(tmp_path / 'migrated.db').as_posix()}"
    os.environ["ALEMBIC_DATABASE_URL"] = url
    try:
        cfg = Config(str(ROOT_DIR / "alembic.ini"))
        cfg.set_main_option("script_location", str(ROOT_DIR / "migrations"))
        command.upgrade(cfg, "head")
    finally:
        os.environ.pop("ALEMBIC_DATABASE_URL", None)

    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        migrated = {
            table: {col["name"] for col in inspector.get_columns(table)}
            for table in inspector.get_table_names()
            if table != "alembic_version"
        }
    finally:
        engine.dispose()

    expected = {t.name: set(t.columns.keys()) for t in Base.metadata.sorted_tables}
    assert migrated == expected

"""Account, session and rate-limit logic. No HTTP concerns — the route layer owns those.

The rules this module exists to keep in one place:

* An email address is normalised once, on the way in. Everything downstream compares
  already-normalised values, so case can never split one person into two accounts.
* Authentication failures are indistinguishable. Unknown address and wrong password return
  the same value, take the same time (see :func:`backend.security.verify_password`) and are
  recorded in the same ledger.
* Anything that changes a password ends every other session for that user. A password
  change is what someone does *because* they think a session is compromised; leaving the
  others live would defeat the point.
* Rate limits are counted in the database, per address and per source address, so a limit
  survives a restart and holds across workers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DbSession

from backend.security import (
    hash_password,
    mint_token,
    token_digest,
    verify_password,
)
from backend.settings import settings
from backend.tables import LoginAttempt, PasswordReset, User, UserSession, utcnow

# How stale `last_seen_at` may get before a request writes it back. Without this, every
# authenticated request issues an UPDATE, which on SQLite means taking the write lock for
# a value nothing reads at that precision.
_LAST_SEEN_RESOLUTION = timedelta(minutes=5)


@dataclass(frozen=True)
class Limit:
    kind: str
    window: timedelta
    max_per_email: int | None
    max_per_ip: int | None
    #: When true, successful attempts count too (signup and reset requests always "succeed",
    #: so counting only failures would make the limit unreachable).
    count_successes: bool = False


LIMITS: dict[str, Limit] = {
    "login": Limit("login", timedelta(minutes=15), max_per_email=5, max_per_ip=20),
    "signup": Limit("signup", timedelta(hours=1), max_per_email=None, max_per_ip=5,
                    count_successes=True),
    "reset": Limit("reset", timedelta(hours=1), max_per_email=3, max_per_ip=10,
                   count_successes=True),
}


class RateLimited(Exception):
    def __init__(self, retry_after_seconds: int):
        super().__init__("Too many attempts.")
        self.retry_after_seconds = retry_after_seconds


class EmailAlreadyRegistered(Exception):
    pass


# --- normalisation -----------------------------------------------------------------------


def normalize_email(email: str) -> str:
    return email.strip().lower()


# --- rate limiting -----------------------------------------------------------------------


def _count(db: DbSession, limit: Limit, since: datetime, *, email: str | None, ip: str | None) -> int:
    stmt = select(func.count()).select_from(LoginAttempt).where(
        LoginAttempt.kind == limit.kind, LoginAttempt.created_at >= since
    )
    if not limit.count_successes:
        stmt = stmt.where(LoginAttempt.succeeded.is_(False))
    if email is not None:
        stmt = stmt.where(LoginAttempt.email == email)
    else:
        stmt = stmt.where(LoginAttempt.ip == ip)
    return int(db.execute(stmt).scalar_one())


def check_rate_limit(db: DbSession, kind: str, *, email: str | None, ip: str | None) -> None:
    """Raise :class:`RateLimited` if this address or source has had too many recent tries."""
    limit = LIMITS[kind]
    since = utcnow() - limit.window
    retry_after = int(limit.window.total_seconds())

    if limit.max_per_email is not None and email:
        if _count(db, limit, since, email=email, ip=None) >= limit.max_per_email:
            raise RateLimited(retry_after)
    if limit.max_per_ip is not None and ip:
        if _count(db, limit, since, email=None, ip=ip) >= limit.max_per_ip:
            raise RateLimited(retry_after)


def record_attempt(
    db: DbSession, kind: str, *, email: str | None, ip: str | None, succeeded: bool
) -> None:
    db.add(LoginAttempt(kind=kind, email=email, ip=ip, succeeded=succeeded))
    db.commit()


def clear_failed_logins(db: DbSession, email: str) -> None:
    """Called on a successful sign-in so an earlier fumble does not accumulate toward a lockout."""
    db.execute(
        delete(LoginAttempt).where(
            LoginAttempt.kind == "login",
            LoginAttempt.email == email,
            LoginAttempt.succeeded.is_(False),
        )
    )
    db.commit()


# --- accounts ----------------------------------------------------------------------------


def get_user_by_email(db: DbSession, email: str) -> User | None:
    return db.execute(select(User).where(User.email == normalize_email(email))).scalar_one_or_none()


def create_user(db: DbSession, *, email: str, password: str, display_name: str | None) -> User:
    normalized = normalize_email(email)
    user = User(
        email=normalized,
        display_name=(display_name or "").strip() or None,
        password_hash=hash_password(password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        # The unique index is the authority, not a prior SELECT: two simultaneous signups
        # for the same address both pass a check-then-insert.
        db.rollback()
        raise EmailAlreadyRegistered(normalized) from exc
    db.refresh(user)
    return user


def authenticate(db: DbSession, *, email: str, password: str) -> User | None:
    """The user, or ``None``. Identical cost and result shape for both failure modes."""
    user = get_user_by_email(db, email)
    ok, refreshed = verify_password(user.password_hash if user else None, password)
    if not ok or user is None:
        return None
    if refreshed:
        user.password_hash = refreshed
        db.commit()
    return user


def set_password(db: DbSession, user: User, new_password: str, *, keep_session_id: str | None) -> None:
    """Change the password and drop every other session."""
    user.password_hash = hash_password(new_password)
    stmt = delete(UserSession).where(UserSession.user_id == user.id)
    if keep_session_id:
        stmt = stmt.where(UserSession.id != keep_session_id)
    db.execute(stmt)
    db.commit()


def delete_user(db: DbSession, user: User) -> None:
    """Hard delete. Sessions, resets and saved analyses go with it, by cascade."""
    db.delete(user)
    db.commit()


# --- sessions ----------------------------------------------------------------------------


def create_session(
    db: DbSession, user: User, *, user_agent: str | None, ip: str | None
) -> tuple[str, UserSession]:
    """Create a session row. Returns the **raw** token (for the cookie) and the row.

    Both are returned because the caller needs the expiry and the raw token, and looking
    the row back up by "most recently created for this user" would pick the wrong one when
    two sign-ins land in the same instant.
    """
    raw = mint_token()
    session = UserSession(
        user_id=user.id,
        token_hash=token_digest(raw),
        expires_at=utcnow() + timedelta(days=settings.session_ttl_days),
        user_agent=(user_agent or "")[:255] or None,
        ip=ip,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return raw, session


def resolve_session(db: DbSession, raw_token: str) -> tuple[User, UserSession] | None:
    """Look a cookie up. Expired or idle-timed-out sessions are deleted, not just refused."""
    session = db.execute(
        select(UserSession).where(UserSession.token_hash == token_digest(raw_token))
    ).scalar_one_or_none()
    if session is None:
        return None

    now = utcnow()
    idle_cutoff = now - timedelta(days=settings.session_idle_days)
    if session.expires_at <= now or session.last_seen_at <= idle_cutoff:
        db.delete(session)
        db.commit()
        return None

    if now - session.last_seen_at > _LAST_SEEN_RESOLUTION:
        session.last_seen_at = now
        db.commit()

    user = db.get(User, session.user_id)
    if user is None:  # pragma: no cover - cascade makes this unreachable
        db.delete(session)
        db.commit()
        return None
    return user, session


def revoke_session(db: DbSession, session: UserSession) -> None:
    db.delete(session)
    db.commit()


def revoke_other_sessions(db: DbSession, user: User, keep_session_id: str) -> int:
    result = db.execute(
        delete(UserSession).where(
            UserSession.user_id == user.id, UserSession.id != keep_session_id
        )
    )
    db.commit()
    return int(result.rowcount or 0)


# --- password reset ----------------------------------------------------------------------


def create_password_reset(db: DbSession, user: User) -> str:
    """Invalidate any outstanding grant for this user, then issue one. Returns the raw token."""
    db.execute(delete(PasswordReset).where(PasswordReset.user_id == user.id))
    raw = mint_token()
    db.add(
        PasswordReset(
            user_id=user.id,
            token_hash=token_digest(raw),
            expires_at=utcnow() + timedelta(minutes=settings.reset_ttl_minutes),
        )
    )
    db.commit()
    return raw


def consume_password_reset(db: DbSession, raw_token: str, new_password: str) -> User | None:
    """Spend a reset token. Returns the user, or ``None`` if the grant is not usable."""
    grant = db.execute(
        select(PasswordReset).where(PasswordReset.token_hash == token_digest(raw_token))
    ).scalar_one_or_none()
    if grant is None or grant.used_at is not None or grant.expires_at <= utcnow():
        return None

    user = db.get(User, grant.user_id)
    if user is None:  # pragma: no cover - cascade makes this unreachable
        return None

    grant.used_at = utcnow()
    user.password_hash = hash_password(new_password)
    # Every session ends: a reset is the recovery path for an account believed compromised.
    db.execute(delete(UserSession).where(UserSession.user_id == user.id))
    db.commit()
    return user


# --- housekeeping ------------------------------------------------------------------------


def sweep_expired(db: DbSession) -> dict[str, int]:
    """Delete what has aged out. Called at startup; safe to call at any time."""
    now = utcnow()
    sessions = db.execute(delete(UserSession).where(UserSession.expires_at <= now)).rowcount
    resets = db.execute(delete(PasswordReset).where(PasswordReset.expires_at <= now)).rowcount
    attempts = db.execute(
        delete(LoginAttempt).where(LoginAttempt.created_at <= now - timedelta(days=2))
    ).rowcount
    db.commit()
    return {
        "sessions": int(sessions or 0),
        "password_resets": int(resets or 0),
        "login_attempts": int(attempts or 0),
    }

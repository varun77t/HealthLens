"""ORM tables for accounts, sessions and password resets.

Named ``tables`` rather than ``models`` on purpose — see the note in :mod:`backend.db`.

Three conventions hold across every table here:

* **Timestamps are naive UTC.** SQLite does not preserve an offset, so a mix of aware and
  naive values would compare fine on Postgres and raise ``TypeError`` on SQLite — usually
  in the expiry check, which is exactly the code that must not break. :func:`utcnow` is the
  only way a timestamp is produced.
* **No secret is stored in a form that can be replayed.** Passwords are argon2id digests;
  session and reset tokens are stored as SHA-256 digests of a 256-bit random value. A
  reader of this database cannot sign in as anyone.
* **Deletion is real.** ``ON DELETE CASCADE`` on every user-owned row, and the account
  route issues a hard ``DELETE``. A soft-delete flag would leave the email address and the
  saved health values in the table after the user asked for them to be removed, which is
  not what the interface promises.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base


def utcnow() -> datetime:
    """Current UTC time, naive. The single source of timestamps in this application."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_id() -> str:
    return uuid.uuid4().hex


def as_utc(value: datetime) -> datetime:
    """Re-attach UTC for serialization, so the API emits an unambiguous instant."""
    return value.replace(tzinfo=timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    # Stored already normalised (stripped, lowercased) by the auth service, so a unique
    # index is enough to stop "A@b.com" and "a@b.com" becoming two accounts.
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String(60), nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    sessions: Mapped[list["UserSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    password_resets: Mapped[list["PasswordReset"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def greeting_name(self) -> str:
        """What "Welcome back, —" should say. Never the full email address."""
        if self.display_name and self.display_name.strip():
            return self.display_name.strip()
        return self.email.split("@", 1)[0]


class UserSession(Base):
    """One signed-in browser. The raw token exists only in the cookie."""

    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    # Absolute expiry. Reached regardless of activity, so a forgotten session on a shared
    # machine does not live forever.
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    # Idle expiry is derived from this at check time.
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    user_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ip: Mapped[str | None] = mapped_column(String(45), nullable=True)  # 45 = max IPv6 text

    user: Mapped[User] = relationship(back_populates="sessions")


class PasswordReset(Base):
    """A single-use, short-lived reset grant."""

    __tablename__ = "password_resets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[User] = relationship(back_populates="password_resets")


class LoginAttempt(Base):
    """Rate-limiting ledger.

    In the table rather than in process memory so the limit survives a reload and still
    holds with more than one worker — an in-memory counter is a limit that disappears
    exactly when someone is hammering the process hard enough to restart it.

    ``email`` is recorded even when no such account exists, because that is precisely the
    case a credential-stuffing run produces. It is normalised, never hashed: this table is
    the reason a lockout can be explained, and an opaque one could not be.
    """

    __tablename__ = "login_attempts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    # "login", "signup" or "reset" — one ledger, so a single sweep prunes all of them.
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    ip: Mapped[str | None] = mapped_column(String(45), nullable=True)
    succeeded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


Index("ix_login_attempts_email_created", LoginAttempt.email, LoginAttempt.created_at)
Index("ix_login_attempts_ip_created", LoginAttempt.ip, LoginAttempt.created_at)

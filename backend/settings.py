"""Runtime configuration, read from the environment once at import.

Deployment-shaped settings only. Anything that must stay consistent between the ML
pipeline, the notebooks and the API (paths, the random seed, the disease registry, the
disclaimer) stays in the top-level :mod:`config` module; this file holds the things that
legitimately differ between a laptop and a server.

**There is no signing secret, and that is deliberate.** Session and password-reset tokens
are 256 bits from :func:`secrets.token_urlsafe`, stored only as SHA-256 digests. Nothing is
signed, so there is no key whose leak would forge a session, and no key to rotate. The
alternative — a JWT signed with an ``APP_SECRET`` — would add a secret to protect in
exchange for making sessions unrevocable, which is a bad trade for an application holding
health information.

Production is not allowed to inherit development defaults for anything that matters:
:func:`validate` refuses to start when ``ENV=production`` and the origin list, cookie
policy or database are still the local ones.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from config import DATA_DIR

_TRUE = {"1", "true", "yes", "on"}


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in _TRUE


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def _default_database_url() -> str:
    # SQLite, in data/ alongside the cached datasets. `data/*.db` is gitignored: the
    # development database holds real passwords hashes and real health values typed in
    # while testing, and must never be committed.
    return f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}"


@dataclass(frozen=True)
class Settings:
    env: str
    database_url: str
    allowed_origins: list[str]
    cookie_name: str
    cookie_secure: bool
    cookie_samesite: str
    session_ttl_days: int
    session_idle_days: int
    reset_ttl_minutes: int
    app_base_url: str
    docs_enabled: bool
    smtp_host: str | None
    smtp_port: int
    smtp_user: str | None
    smtp_password: str | None
    smtp_from: str
    smtp_starttls: bool
    trust_proxy_headers: bool = False
    # Origins the ALLOWED_ORIGINS default covers, kept for the production check.
    _dev_origins: tuple[str, ...] = field(default=("http://localhost:5173",))

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host)

    @property
    def session_ttl_seconds(self) -> int:
        return self.session_ttl_days * 24 * 3600


def load() -> Settings:
    env = (os.getenv("ENV") or "development").strip().lower()
    origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
    return Settings(
        env=env,
        database_url=os.getenv("DATABASE_URL") or _default_database_url(),
        allowed_origins=[o.strip() for o in origins_raw.split(",") if o.strip()],
        cookie_name=os.getenv("SESSION_COOKIE_NAME", "mda_session"),
        # Secure cookies are the default everywhere except development, where the dev
        # server is plain http and a Secure cookie would simply never be stored.
        cookie_secure=_flag("COOKIE_SECURE", env == "production"),
        cookie_samesite=os.getenv("COOKIE_SAMESITE", "lax").strip().lower(),
        session_ttl_days=_int("SESSION_TTL_DAYS", 14),
        session_idle_days=_int("SESSION_IDLE_DAYS", 7),
        reset_ttl_minutes=_int("PASSWORD_RESET_TTL_MINUTES", 30),
        app_base_url=os.getenv("APP_BASE_URL", "http://localhost:5173").rstrip("/"),
        docs_enabled=_flag("DOCS_ENABLED", env != "production"),
        smtp_host=os.getenv("SMTP_HOST") or None,
        smtp_port=_int("SMTP_PORT", 587),
        smtp_user=os.getenv("SMTP_USER") or None,
        smtp_password=os.getenv("SMTP_PASSWORD") or None,
        smtp_from=os.getenv("SMTP_FROM", "no-reply@multi-disease-ai.local"),
        smtp_starttls=_flag("SMTP_STARTTLS", True),
        # Off by default. X-Forwarded-For is client-controlled unless a trusted proxy
        # overwrites it, and believing it would let one host defeat every per-IP rate
        # limit by inventing a new address per request.
        trust_proxy_headers=_flag("TRUST_PROXY_HEADERS", False),
    )


def validate(s: Settings) -> list[str]:
    """Configuration problems that should stop a production start. Empty list = fine."""
    problems: list[str] = []
    if s.cookie_samesite not in {"lax", "strict", "none"}:
        problems.append(
            f"COOKIE_SAMESITE must be lax, strict or none — got {s.cookie_samesite!r}."
        )
    if s.cookie_samesite == "none" and not s.cookie_secure:
        problems.append("COOKIE_SAMESITE=none requires COOKIE_SECURE=true (browsers reject it otherwise).")
    if not s.allowed_origins:
        problems.append("ALLOWED_ORIGINS is empty; the frontend would be blocked by CORS.")
    if "*" in s.allowed_origins:
        problems.append(
            "ALLOWED_ORIGINS cannot be '*' — credentials are enabled, and browsers reject "
            "a wildcard origin on a credentialed request anyway."
        )
    if not s.is_production:
        return problems

    if not s.cookie_secure:
        problems.append("COOKIE_SECURE must be true in production; the session cookie would be sent over http.")
    if any(o.startswith("http://") for o in s.allowed_origins):
        problems.append(f"ALLOWED_ORIGINS contains a plain-http origin in production: {s.allowed_origins}.")
    if os.getenv("ALLOWED_ORIGINS") is None:
        problems.append("ALLOWED_ORIGINS must be set explicitly in production, not inherited from the dev default.")
    if s.app_base_url.startswith("http://"):
        problems.append("APP_BASE_URL must be https in production; password-reset links are built from it.")
    if s.is_sqlite:
        problems.append(
            "DATABASE_URL is still SQLite in production. SQLite has no at-rest encryption "
            "and one writer; set a Postgres URL."
        )
    return problems


settings = load()


def database_path() -> Path | None:
    """Filesystem location of the SQLite database, or None for a server database."""
    if not settings.is_sqlite:
        return None
    return Path(settings.database_url.replace("sqlite:///", "", 1))

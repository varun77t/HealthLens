"""Database engine, session factory and the FastAPI dependency that hands out sessions.

The ORM tables live in :mod:`backend.tables` rather than a ``backend/models/`` package,
because ``models/`` already means trained ML artifacts everywhere else in this repository
and ``backend/routes/models.py`` already serves model cards. One more meaning of the word
would make the imports actively misleading.

Two SQLite-specific details are handled here, both of which are silent data bugs if
forgotten:

* ``PRAGMA foreign_keys=ON`` is issued on every connection. SQLite ignores
  ``ON DELETE CASCADE`` unless it is set, per connection — so "delete my account" would
  leave the sessions and saved analyses behind while appearing to succeed.
* ``check_same_thread=False``, because FastAPI runs sync endpoints in a threadpool and a
  connection is not guaranteed to be reused on the thread that created it.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.settings import settings


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _connect_args() -> dict:
    return {"check_same_thread": False} if settings.is_sqlite else {}


def engine() -> Engine:
    global _engine
    if _engine is None:
        url = settings.database_url
        if settings.is_sqlite:
            path = Path(url.replace("sqlite:///", "", 1))
            path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(url, connect_args=_connect_args(), future=True)
        if settings.is_sqlite:
            @event.listens_for(_engine, "connect")
            def _enable_foreign_keys(dbapi_connection, _record):  # pragma: no cover - trivial
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
    return _engine


def session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=engine(), autoflush=False, expire_on_commit=False)
    return _session_factory


def reset_engine() -> None:
    """Drop the cached engine so a test can point ``DATABASE_URL`` somewhere else."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    db = session_factory()()
    try:
        yield db
    finally:
        db.close()


def create_all() -> None:
    """Create every table directly, bypassing Alembic.

    Used by the test suite, which builds a throwaway database per run. Application startup
    does **not** call this: schema changes go through ``alembic upgrade head`` so that a
    development database keeps its rows across a migration.
    """
    import backend.tables  # noqa: F401  -- registers the mappings on Base.metadata

    Base.metadata.create_all(bind=engine())

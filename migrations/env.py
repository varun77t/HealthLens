"""Alembic environment.

The database URL comes from :mod:`backend.settings` (and therefore from ``DATABASE_URL``),
not from ``alembic.ini``. One source of truth means a migration cannot be applied to a
different database than the application uses — which is the sort of mistake that is only
noticed after the application has been running against an un-migrated schema.

``render_as_batch`` is on because the development database is SQLite, which cannot
``ALTER TABLE ... DROP COLUMN`` or alter a constraint. Batch mode makes Alembic rebuild the
table instead, so the same migration script runs on both SQLite and Postgres.
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from backend.db import Base
from backend.settings import settings

# Import for the side effect of registering every mapping on Base.metadata.
import backend.tables  # noqa: F401

# ALEMBIC_DATABASE_URL is an explicit override, used by the test that applies the
# migrations to a scratch database and compares the result against the ORM metadata.
# Without it that test would migrate the very database the suite is already using.
DATABASE_URL = os.getenv("ALEMBIC_DATABASE_URL") or settings.database_url

config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

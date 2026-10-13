"""
Alembic migration environment.

This file is executed by Alembic for every migration command.
It handles both offline (SQL script generation) and online (live DB) modes.

Async PostgreSQL support:
    SQLAlchemy's async engine cannot be used directly by Alembic's synchronous
    migration runner. The workaround is to create an async engine, open an async
    connection, and then call `connection.run_sync(do_run_migrations)` — this
    hands a synchronous connection handle to Alembic while keeping the event loop.

Database URL:
    Read from application settings (which reads from the .env file at the
    monorepo root). This keeps credentials in a single place.
"""
from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig

from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

from alembic import context

# Ensure backend/ (the project root) is in sys.path so we can import our modules.
# alembic/env.py lives at: backend/alembic/env.py
# We go one level up to reach backend/.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import settings first so the URL is available
# Import all models to register them with Base.metadata.
# Without this import, Alembic autogenerate will produce empty migrations.
import db.models  # noqa: F401, E402
from core.config import get_settings  # noqa: E402
from db.models.base import Base  # noqa: E402

# Alembic Config object — provides access to alembic.ini values
config = context.config

# Configure stdlib logging from the alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The metadata object Alembic uses to detect schema changes
target_metadata = Base.metadata


def get_url() -> str:
    """Return the async PostgreSQL URL from application settings."""
    return get_settings().database_url


# ── Offline mode: generates SQL without connecting ────────────────────────────

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    Generates SQL that can be piped to psql instead of running directly.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode: connects and runs migrations live ────────────────────────────

def do_run_migrations(connection) -> None:
    """Execute migrations using a synchronous connection handle (from run_sync)."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,   # Detect column type changes
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Create an async engine and run migrations inside a synchronous callback.
    NullPool is used so Alembic doesn't hold connections between migrations.
    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migration mode — runs the async function."""
    asyncio.run(run_async_migrations())


# ── Dispatch ──────────────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

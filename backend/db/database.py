"""
Database engine, session management, and the DatabaseManager singleton.

Design:
- DatabaseManager is a singleton (module-level `db_manager`).
- connect() is called once at application startup in the lifespan handler.
- disconnect() is called at shutdown to cleanly drain the connection pool.
- get_session() is an async generator used as a FastAPI dependency.
- check_connectivity() is a lightweight ping used by the /ready endpoint.

Usage as a FastAPI dependency:
    from db.database import db_manager
    from fastapi import Depends
    from sqlalchemy.ext.asyncio import AsyncSession

    async def some_endpoint(db: AsyncSession = Depends(db_manager.get_session)):
        result = await db.execute(...)
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

log = structlog.get_logger(__name__)


class DatabaseManager:
    """
    Manages the SQLAlchemy async engine lifecycle and provides session dependencies.

    This class is intended to be used as a module-level singleton (`db_manager`).
    It must be connected (connect()) before use and disconnected (disconnect())
    during application shutdown.
    """

    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def connect(self) -> None:
        """
        Initialise the async engine and session factory.
        Called once during application startup.
        """
        from core.config import get_settings
        settings = get_settings()

        self._engine = create_async_engine(
            settings.database_url,
            pool_size=settings.postgres_pool_size,
            max_overflow=settings.postgres_max_overflow,
            pool_timeout=settings.postgres_pool_timeout,
            pool_pre_ping=True,   # Verify connections before checkout — prevents stale connections
            echo=False,           # Set to True only for debugging SQL; too noisy otherwise
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Avoid lazy-load errors after commit
        )
        log.info(
            "Database engine initialised",
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
        )

    async def disconnect(self) -> None:
        """
        Dispose of the engine and close all pool connections.
        Called once during application shutdown.
        """
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            log.info("Database engine disposed")

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        FastAPI dependency that yields an async database session.

        The session is automatically committed on success and rolled back on error.
        Use with:
            async def endpoint(db: AsyncSession = Depends(db_manager.get_session)):
        """
        if self._session_factory is None:
            await self.connect()

        assert self._session_factory is not None
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def check_connectivity(self) -> tuple[bool, str]:
        """
        Perform a lightweight connectivity check against the database.
        Returns (True, "ok") on success, (False, error_message) on failure.
        Used by the /ready health endpoint.
        """
        if self._engine is None:
            try:
                await self.connect()
            except Exception as e:
                return False, f"Database engine not initialised: {e}"

        assert self._engine is not None
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True, "ok"
        except Exception as exc:
            log.warning("Database connectivity check failed", error=str(exc))
            return False, str(exc)


# Module-level singleton — imported by the lifespan handler and API dependencies
db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async for session in db_manager.get_session():
        yield session


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the active session factory or a fresh standalone one."""
    if db_manager._session_factory is not None:
        return db_manager._session_factory
    from core.config import get_settings
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


"""
AI Research Copilot — FastAPI Application Entry Point.

Run with:
    uvicorn main:app --reload --port 8000

Application factory pattern:
    `create_app()` builds and configures the FastAPI application.
    `app` is the module-level instance used by uvicorn.

Lifespan:
    Startup: configure logging → connect database
    Shutdown: disconnect database

Exception handling:
    AppException subclasses → structured JSON with appropriate status code
    Unhandled exceptions    → 500 with generic message (detail NOT exposed)
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.deps import aclose_services
from api.middleware.timing import TimingMiddleware
from api.v1.router import api_v1_router
from core.config import get_settings
from core.error_handling import register_exception_handlers
from core.logging import setup_logging
from db.database import db_manager

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Everything before `yield` runs at startup.
    Everything after `yield` runs at shutdown.
    """
    settings = get_settings()

    # Logging must be configured first so all subsequent messages are formatted
    setup_logging(log_level=settings.app_log_level, is_development=settings.is_development)

    log.info(
        "Starting AI Research Copilot",
        version=app.version,
        environment=settings.app_env,
        debug=settings.app_debug,
    )

    # Establish the database connection pool
    await db_manager.connect()

    log.info("Startup complete — API is ready to accept requests")

    yield  # ← Application is live and serving requests

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("Shutting down AI Research Copilot")
    await aclose_services()  # close retriever/indexer network clients
    await db_manager.disconnect()
    log.info("Shutdown complete")


def create_app() -> FastAPI:
    """
    Application factory.

    Creates, configures, and returns a FastAPI application instance.
    Called once at module load time (for uvicorn) and in tests.
    """
    settings = get_settings()

    app = FastAPI(
        title="AI Research Copilot API",
        description=(
            "AI-powered research and search system. "
            "Investigates complex questions across multiple information sources "
            "using hybrid retrieval, reranking, and grounded LLM synthesis."
        ),
        version="0.1.0",
        contact={
            "name": "AI Research Copilot",
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    # Order matters: middleware is applied in reverse registration order.
    # TimingMiddleware wraps everything so it measures total request time.
    app.add_middleware(TimingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(api_v1_router, prefix="/api/v1")

    # ── Exception Handlers (sanitized; detail stays in redacted server logs) ──
    register_exception_handlers(app)

    return app


# Module-level application instance consumed by uvicorn and test clients
app = create_app()

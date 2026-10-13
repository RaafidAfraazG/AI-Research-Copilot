"""
Pytest configuration and shared fixtures.

Test strategy:
- Unit tests use mocks for all external I/O (DB, OpenSearch, Qdrant, Redis).
- Model tests use an in-memory SQLite database via aiosqlite (no Docker needed).
- API tests use httpx.AsyncClient with ASGITransport and mocked infrastructure.
- No real API keys or external services are required to run the test suite.

Environment:
- The test suite runs with POSTGRES_PASSWORD set to avoid Settings validation errors.
- get_settings() is cleared between tests that patch env vars.
"""
from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ── Bootstrap: ensure required env vars exist before any module is imported ───
os.environ.setdefault("POSTGRES_PASSWORD", "test_password_for_ci")
os.environ.setdefault("APP_ENV", "development")

# Import models AFTER env vars are set so Settings loads cleanly
from db.models.base import Base  # noqa: E402

# ── Settings cache management ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the lru_cache on get_settings() around every test."""
    from core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── Data-root isolation & embedding configuration ─────────────────────────────

@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """
    Every test gets its own canonical data root. Tests must never read or write the
    real repository index (Phase 8.6: this is how test fixtures once polluted it).
    """
    from core.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    yield tmp_path / "data"


@pytest.fixture(autouse=True)
def reset_embedding_provider_cache():
    """Per-test isolation of process-level singletons (embedder, API services, reranker)."""
    from api.deps import reset_services
    from embedding.factory import get_embedding_provider
    from ranking.ranking_service import reset_shared_rerankers

    get_embedding_provider.cache_clear()
    reset_services()
    reset_shared_rerankers()
    yield
    get_embedding_provider.cache_clear()
    reset_services()
    reset_shared_rerankers()


MOCK_CORPUS = [
    {
        "source_type": "arxiv",
        "source_id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "content": "Retrieval-augmented generation combines parametric memory with a non-parametric "
        "dense passage index to answer knowledge-intensive questions.",
        "metadata": {"year": 2020},
    },
    {
        "source_type": "arxiv",
        "source_id": "2004.04906",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "content": "Dense passage retrieval uses a dual encoder to embed questions and passages "
        "for open-domain question answering.",
        "metadata": {"year": 2020},
    },
    {
        "source_type": "arxiv",
        "source_id": "1706.03762",
        "title": "Attention Is All You Need",
        "content": "The Transformer relies entirely on self-attention mechanisms instead of recurrence.",
        "metadata": {"year": 2017},
    },
]


@pytest.fixture
def mock_embeddings_configured(monkeypatch):
    """Explicitly opt in to the mock embedding provider for this test."""
    from core.config import get_settings
    from embedding.factory import get_embedding_provider

    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    get_settings.cache_clear()
    get_embedding_provider.cache_clear()
    yield


@pytest.fixture
def mock_index(mock_embeddings_configured):
    """A complete, active index built with the EXPLICITLY configured mock embedder."""
    from core.config import get_settings
    from embedding.factory import get_embedding_provider
    from retrieval.index_admin import rebuild_index
    from retrieval.index_manifest import IndexStore

    store = IndexStore(settings=get_settings())
    report = rebuild_index(
        store, get_embedding_provider(), MOCK_CORPUS, settings=get_settings(), allow_mock=True
    )
    assert report["status"] == "complete", report
    return store


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_model: needs the real embedding/cross-encoder weights (opt-in: RUN_REAL_MODEL_TESTS=1)"
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_REAL_MODEL_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="real-model test; set RUN_REAL_MODEL_TESTS=1 to run")
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip)


# ── In-memory SQLite database (model tests) ───────────────────────────────────

@pytest.fixture
async def test_engine():
    """
    Create an in-memory SQLite engine with all ORM tables.
    Does NOT require Docker or a running PostgreSQL instance.
    """
    # aiosqlite driver for async SQLite
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Async session bound to the in-memory test database.
    Rolls back after each test for isolation.
    """
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()


# ── Infrastructure mocks (API tests) ─────────────────────────────────────────

@pytest.fixture
def mock_infra():
    """
    Patch all four infrastructure check functions in health.py to return
    CheckResult(status="ok"). Tests that need failure scenarios can override
    individual mock return values after receiving this fixture.
    """
    from api.v1.health import CheckResult

    ok = CheckResult(status="ok", latency_ms=1.0)

    with (
        patch("api.v1.health._check_postgres", new_callable=AsyncMock) as pg,
        patch("api.v1.health._check_opensearch", new_callable=AsyncMock) as osr,
        patch("api.v1.health._check_qdrant", new_callable=AsyncMock) as qd,
        patch("api.v1.health._check_redis", new_callable=AsyncMock) as rd,
        patch("api.v1.health._check_index", new_callable=AsyncMock) as ix,
        patch("api.v1.health._check_llm", new_callable=AsyncMock) as lm,
    ):
        pg.return_value = ok
        osr.return_value = ok
        qd.return_value = ok
        rd.return_value = ok
        ix.return_value = ok
        lm.return_value = ok
        yield {
            "postgres": pg, "opensearch": osr, "qdrant": qd, "redis": rd, "index": ix, "llm": lm,
        }


# ── Async HTTP client (API tests) ─────────────────────────────────────────────

@pytest.fixture
async def api_client(mock_infra) -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP test client with:
    - All infrastructure checks mocked (via mock_infra)
    - Database connect/disconnect mocked (lifespan doesn't touch real DB)
    - Full FastAPI middleware stack active
    """
    from db import database as db_module
    from main import app

    with (
        patch.object(db_module.db_manager, "connect", new_callable=AsyncMock),
        patch.object(db_module.db_manager, "disconnect", new_callable=AsyncMock),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            yield client

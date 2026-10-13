"""
Health and readiness endpoints.

GET /api/v1/health  — Liveness probe. Always 200 while the API process is alive.
GET /api/v1/ready   — Readiness probe. 200 if all required infrastructure is reachable,
                      503 if any service is down.

Design notes:
- /health is a simple liveness check — no I/O, instant response.
- /ready creates short-lived clients per call rather than reusing the
  application-level connection pool. This keeps the check isolated.
- All infrastructure checks run concurrently via asyncio.gather().
- Each check has a hard timeout (HEALTH_CHECK_TIMEOUT_S) to avoid hanging.
- Errors are caught and recorded per-service; the endpoint never raises 500.
"""
from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any, Literal

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = structlog.get_logger(__name__)
router = APIRouter(tags=["Health"])

# Hard per-service timeout for readiness checks
HEALTH_CHECK_TIMEOUT_S: float = 3.0


# ── Response schemas ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    environment: str
    timestamp: datetime


class CheckResult(BaseModel):
    status: Literal["ok", "error", "skipped"]
    latency_ms: float | None = None
    error: str | None = None
    # Only required checks decide readiness. Optional/skipped ones are informational.
    required: bool = True
    note: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckResult]
    timestamp: datetime


# ── Per-service check helpers ─────────────────────────────────────────────────

def _safe_error(exc: BaseException) -> str:
    """Client-safe error description: exception class only; detail goes to logs (redacted)."""
    log.warning("readiness_check_failed", error_type=type(exc).__name__, error=str(exc))
    return type(exc).__name__


async def _check_postgres() -> CheckResult:
    """Ping PostgreSQL with a SELECT 1. Uses a disposable one-shot engine."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from core.config import get_settings

    settings = get_settings()
    start = time.perf_counter()
    engine = None
    try:
        async with asyncio.timeout(HEALTH_CHECK_TIMEOUT_S):
            engine = create_async_engine(
                settings.database_url,
                pool_size=1,
                max_overflow=0,
                pool_timeout=int(HEALTH_CHECK_TIMEOUT_S),
            )
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        return CheckResult(
            status="ok",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
    except Exception as exc:
        return CheckResult(
            status="error",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            error=_safe_error(exc),
        )
    finally:
        if engine is not None:
            try:
                await engine.dispose()
            except Exception:
                pass


async def _check_opensearch() -> CheckResult:
    """Ping OpenSearch cluster health endpoint."""
    from opensearchpy import AsyncOpenSearch

    from core.config import get_settings

    settings = get_settings()
    start = time.perf_counter()
    client = None
    try:
        async with asyncio.timeout(HEALTH_CHECK_TIMEOUT_S):
            http_auth = (
                (settings.opensearch_user, settings.opensearch_password)
                if settings.opensearch_user
                else None
            )
            client = AsyncOpenSearch(
                hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
                use_ssl=settings.opensearch_use_ssl,
                verify_certs=False,
                http_auth=http_auth,
                timeout=int(HEALTH_CHECK_TIMEOUT_S),
            )
            await client.cluster.health()
        return CheckResult(
            status="ok",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
    except Exception as exc:
        return CheckResult(
            status="error",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            error=_safe_error(exc),
        )
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass


async def _check_qdrant() -> CheckResult:
    """Ping Qdrant by listing collections."""
    from qdrant_client import AsyncQdrantClient

    from core.config import get_settings

    settings = get_settings()
    start = time.perf_counter()
    client = None
    try:
        async with asyncio.timeout(HEALTH_CHECK_TIMEOUT_S):
            client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                timeout=int(HEALTH_CHECK_TIMEOUT_S),
            )
            await client.get_collections()
        return CheckResult(
            status="ok",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
    except Exception as exc:
        return CheckResult(
            status="error",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            error=_safe_error(exc),
        )
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass


async def _check_redis() -> CheckResult:
    """Ping Redis."""
    import redis.asyncio as aioredis

    from core.config import get_settings

    settings = get_settings()
    start = time.perf_counter()
    client = None
    try:
        async with asyncio.timeout(HEALTH_CHECK_TIMEOUT_S):
            client = aioredis.from_url(
                settings.redis_url,
                socket_connect_timeout=int(HEALTH_CHECK_TIMEOUT_S),
                socket_timeout=int(HEALTH_CHECK_TIMEOUT_S),
                decode_responses=True,
            )
            await client.ping()
        return CheckResult(
            status="ok",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
    except Exception as exc:
        return CheckResult(
            status="error",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            error=_safe_error(exc),
        )
    finally:
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass


async def _check_index() -> CheckResult:
    """
    Active vector index must exist, be complete, match the configured embedding
    provider/model, and its stores must be present. Cheap: reads the manifest only
    (the embedding model is NOT loaded here).
    """
    from core.config import get_settings
    from core.exceptions import AppException
    from retrieval.index_manifest import IndexStore

    settings = get_settings()
    start = time.perf_counter()
    try:
        store = IndexStore(settings=settings)
        manifest = store.active_manifest()
        emb = manifest.embedding
        configured = settings.embedding_provider
        expected_kind = {"local": "sentence_transformers", "mock": "mock", "openai": "openai"}.get(
            configured, configured
        )
        if emb.provider != expected_kind or (
            configured != "mock" and emb.model_name != settings.embedding_model
        ):
            return CheckResult(
                status="error",
                error="embedding_model_mismatch",
                note="Index was built with a different embedding model than configured. Rebuild the index.",
            )
        if not store.bm25_path(manifest.generation_id).exists() or not store.qdrant_dir(
            manifest.generation_id
        ).exists():
            return CheckResult(status="error", error="index_files_missing")
        return CheckResult(
            status="ok",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            note=f"{manifest.counts.documents} documents / {manifest.counts.chunks} chunks",
        )
    except AppException as exc:
        return CheckResult(status="error", error=exc.code)
    except Exception as exc:
        return CheckResult(status="error", error=_safe_error(exc))


async def _check_llm() -> CheckResult:
    """LLM provider configuration check. No provider request is made (no quota used)."""
    from core.config import get_settings

    st = get_settings()
    provider = st.llm_provider
    if provider == "mock":
        return CheckResult(status="ok", note="mock provider (not a real model)")
    if provider == "gemini":
        if not st.gemini_api_key:
            return CheckResult(status="error", error="gemini_api_key_not_configured")
        return CheckResult(status="ok", note=f"gemini ({st.gemini_model}), configuration only")
    if provider in {"openai", "anthropic"}:
        if not st.llm_allow_paid_provider:
            return CheckResult(status="error", error="paid_provider_not_allowed")
        key = st.openai_api_key if provider == "openai" else st.anthropic_api_key
        return CheckResult(
            status="ok" if key else "error", error=None if key else "api_key_not_configured"
        )
    return CheckResult(status="ok", note=f"{provider}, configuration only")


async def _optional(check: Any, name: str) -> CheckResult:
    """
    OpenSearch / Qdrant server / Redis are NOT part of the local architecture
    (local BM25 + embedded Qdrant). They are skipped unless READINESS_PROBE_OPTIONAL=true,
    and even then never affect readiness.
    """
    from core.config import get_settings

    if not get_settings().readiness_probe_optional:
        return CheckResult(
            status="skipped",
            required=False,
            note=f"{name} is not used by the local architecture",
        )
    res = await check()
    res.required = False
    return res


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    description=(
        "Returns 200 immediately if the API process is alive. "
        "Does not check infrastructure. Use /ready for dependency checks."
    ),
)
async def health() -> HealthResponse:
    """Liveness probe — no I/O, always fast."""
    from core.config import get_settings
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.app_env,
        timestamp=datetime.now(UTC),
    )


@router.get(
    "/ready",
    summary="Readiness check",
    description=(
        "Readiness for ANSWERING requests: PostgreSQL reachable, the active vector index is "
        "complete and compatible with the configured embedding model, and an LLM provider is "
        "configured. OpenSearch/Qdrant-server/Redis are not part of the local architecture and "
        "are reported as skipped. /health is liveness only."
    ),
    responses={
        200: {"model": ReadinessResponse, "description": "All services ready"},
        503: {"model": ReadinessResponse, "description": "One or more services unavailable"},
    },
)
async def ready() -> Any:
    """
    Readiness probe — runs all infrastructure checks concurrently.

    Required: PostgreSQL, active index integrity/compatibility, LLM configuration.
    Informational (never fail readiness): OpenSearch, Qdrant server, Redis.
    Returns 200 if all required checks pass, 503 otherwise.
    """
    (
        postgres_res,
        index_res,
        llm_res,
        opensearch_res,
        qdrant_res,
        redis_res,
    ) = await asyncio.gather(
        _check_postgres(),
        _check_index(),
        _check_llm(),
        _optional(_check_opensearch, "OpenSearch"),
        _optional(_check_qdrant, "Qdrant server"),
        _optional(_check_redis, "Redis"),
    )

    checks: dict[str, CheckResult] = {
        "postgres": postgres_res,
        "index": index_res,
        "llm": llm_res,
        "opensearch": opensearch_res,
        "qdrant": qdrant_res,
        "redis": redis_res,
    }

    # Only REQUIRED checks decide readiness; optional/skipped services are informational.
    required = {n: r for n, r in checks.items() if r.required}
    all_ok = all(r.status == "ok" for r in required.values())

    if not all_ok:
        failed = [name for name, r in required.items() if r.status != "ok"]
        log.warning("Readiness check: required dependency unavailable", failed=failed)

    response = ReadinessResponse(
        status="ready" if all_ok else "not_ready",
        checks=checks,
        timestamp=datetime.now(UTC),
    )

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content=response.model_dump(mode="json"),
    )

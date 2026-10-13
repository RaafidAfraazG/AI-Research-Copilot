"""
Central, sanitizing exception handling.

Clients receive a stable body ``{"error": <safe message>, "code": <stable code>,
"request_id": <id>}`` with an appropriate HTTP status. Raw exception text, provider
response bodies, stack traces, DSNs and local paths stay in (redacted) server logs and
are correlated with the client via ``request_id``.

Frontend-relevant categories:
    llm_rate_limited (429)   provider quota / rate limit
    llm_timeout / upstream_timeout (504)
    llm_model_unavailable / llm_auth_error / llm_provider_error (502)
    llm_not_configured (503)
    index_* / embedding_* (503)   from AppException subclasses
    url_not_allowed (400)    SSRF validation failure
    internal_error (500)
"""
from __future__ import annotations

import re
import uuid
from typing import Any

import httpx
import structlog
from fastapi import Request
from fastapi.responses import JSONResponse

from core.exceptions import AppException
from core.redaction import redact

log = structlog.get_logger(__name__)

_MALFORMED_RE = re.compile(r"could not be parsed|truncated or malformed", re.I)
_RATE_RE = re.compile(r"429|quota|rate[\s_-]?limit|resource[\s_-]?exhausted|too many requests", re.I)
_TIMEOUT_RE = re.compile(r"timed? ?out|deadline|504", re.I)
_MODEL_RE = re.compile(r"not found|404|unsupported|not supported|no longer available|model.*(unavailable|deprecated)", re.I)
_AUTH_RE = re.compile(r"api key|api_key|permission|unauthori|forbidden|401|403|invalid.*credential|authentication", re.I)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def classify_llm_error(text: str) -> tuple[int, str, str]:
    """Map provider error text to (status, code, safe message). Order matters."""
    if _MALFORMED_RE.search(text):
        return (
            502,
            "llm_malformed_output",
            "The language model returned an incomplete or malformed answer. Try again.",
        )
    if _RATE_RE.search(text):
        return (
            429,
            "llm_rate_limited",
            "The language model is rate limited or its free-tier quota is exhausted. Try again later.",
        )
    if _TIMEOUT_RE.search(text):
        return 504, "llm_timeout", "The language model did not respond in time."
    if _AUTH_RE.search(text):
        return (
            502,
            "llm_auth_error",
            "The language model rejected the configured credentials. Check the backend configuration.",
        )
    if _MODEL_RE.search(text):
        return (
            502,
            "llm_model_unavailable",
            "The configured language model is unavailable. Check GEMINI_MODEL in the backend configuration.",
        )
    return 502, "llm_provider_error", "The language model provider returned an error."


def _body(message: str, code: str, request_id: str) -> dict[str, Any]:
    return {"error": message, "code": code, "request_id": request_id}


def _json(status: int, message: str, code: str, request_id: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=_body(message, code, request_id))


def register_exception_handlers(app: Any) -> None:
    """Attach sanitizing handlers. Detail goes to logs only, never to the client."""
    from fastapi.exceptions import RequestValidationError

    from connectors.url_security import SSRFValidationError
    from llm.models import LLMConfigurationError, LLMError, LLMGenerationError

    @app.exception_handler(AppException)
    async def _app_exc(request: Request, exc: AppException) -> JSONResponse:
        rid = new_request_id()
        log.warning(
            "application_error",
            code=exc.code,
            status=exc.status_code,
            message=redact(exc.message),
            path=request.url.path,
            request_id=rid,
        )
        # AppException messages are authored to be client-safe; still redacted.
        return _json(exc.status_code, redact(exc.message), exc.code, rid)

    @app.exception_handler(LLMConfigurationError)
    async def _llm_config(request: Request, exc: LLMConfigurationError) -> JSONResponse:
        rid = new_request_id()
        log.error("llm_configuration_error", error=str(exc), path=request.url.path, request_id=rid)
        return _json(
            503,
            "The language model is not configured. Check the LLM settings in the backend configuration.",
            "llm_not_configured",
            rid,
        )

    @app.exception_handler(LLMGenerationError)
    async def _llm_gen(request: Request, exc: LLMGenerationError) -> JSONResponse:
        rid = new_request_id()
        status, code, message = classify_llm_error(str(exc))
        log.error("llm_generation_error", code=code, error=str(exc), path=request.url.path, request_id=rid)
        return _json(status, message, code, rid)

    @app.exception_handler(LLMError)
    async def _llm_other(request: Request, exc: LLMError) -> JSONResponse:
        rid = new_request_id()
        log.error("llm_error", error=str(exc), path=request.url.path, request_id=rid)
        return _json(502, "The language model provider returned an error.", "llm_provider_error", rid)

    @app.exception_handler(SSRFValidationError)
    async def _ssrf(request: Request, exc: SSRFValidationError) -> JSONResponse:
        rid = new_request_id()
        log.warning("ssrf_blocked", reason=str(exc), path=request.url.path, request_id=rid)
        return _json(
            400,
            "The URL uses an unsupported scheme or points to a disallowed network destination.",
            "url_not_allowed",
            rid,
        )

    @app.exception_handler(httpx.TimeoutException)
    async def _http_timeout(request: Request, exc: httpx.TimeoutException) -> JSONResponse:
        rid = new_request_id()
        log.warning("upstream_timeout", error_type=type(exc).__name__, path=request.url.path, request_id=rid)
        return _json(504, "An upstream service timed out.", "upstream_timeout", rid)

    @app.exception_handler(httpx.HTTPError)
    async def _http_error(request: Request, exc: httpx.HTTPError) -> JSONResponse:
        rid = new_request_id()
        log.warning("upstream_error", error_type=type(exc).__name__, path=request.url.path, request_id=rid)
        return _json(502, "An upstream service request failed.", "upstream_error", rid)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        rid = new_request_id()
        # Field-level problems are useful and safe; drop echoed input and context objects.
        problems = [
            {"field": ".".join(str(p) for p in e.get("loc", []) if p != "body"), "message": e.get("msg", "invalid")}
            for e in exc.errors()
        ]
        log.info("request_validation_failed", problems=problems, path=request.url.path, request_id=rid)
        content = _body("The request was invalid.", "validation_error", rid)
        content["detail"] = problems
        return JSONResponse(status_code=422, content=content)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        rid = new_request_id()
        log.error(
            "unhandled_exception",
            error_type=type(exc).__name__,
            error=str(exc),
            path=request.url.path,
            request_id=rid,
            exc_info=True,
        )
        return _json(500, "An internal server error occurred.", "internal_error", rid)

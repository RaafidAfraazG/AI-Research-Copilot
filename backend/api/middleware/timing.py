"""
Request timing middleware.

Adds an X-Process-Time-Ms header to every response showing the wall-clock
time in milliseconds from when the request was received to when the response
was ready to send.

This is used for:
- Developer visibility during development
- Latency monitoring in production without a separate APM tool
- Baseline for the per-stage latency tracking added in Phase 8

Note: BaseHTTPMiddleware has a known limitation with streaming responses —
it buffers the full response body. This is acceptable for Phase 1.
If streaming (SSE/WebSocket) is added later, replace with a pure ASGI middleware.
"""
from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class TimingMiddleware(BaseHTTPMiddleware):
    """Adds X-Process-Time-Ms to every HTTP response."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

"""
Resilient Async HTTP Client for External Source Connectors.

Features:
- Configurable timeouts and retries with exponential backoff & jitter
- Automatic handling of HTTP 429 (Rate Limit) and Retry-After headers
- Retry on transient errors (500, 502, 503, 504, connection timeouts)
- Host-level rate limiting / minimum request intervals
- Credential-safe logging (API keys and auth headers are never logged)

Hardened mode (`ssrf_protect=True` and/or `max_bytes=...`), used for user-supplied URLs:
- Redirects are followed MANUALLY (at most `max_redirects`) and every hop, including the
  first request, is validated (scheme, credentials, hostname, all resolved addresses must
  be public). A redirect to a private/loopback/link-local address is rejected.
- The request is sent to the validated IP literal with the original Host header and TLS
  SNI hostname, so the address that was checked is the address that is contacted. This
  narrows the DNS-rebinding window; it is a mitigation, not an absolute guarantee (e.g.
  an HTTP proxy configured via environment would resolve names itself).
- The body is streamed and aborted as soon as it exceeds `max_bytes` (the limit applies to
  decoded bytes, so compression bombs are bounded too). A declared Content-Length above
  the limit is rejected before reading.
- Connect timeout is capped, and a total per-attempt deadline bounds slow responses.
"""
from __future__ import annotations

import asyncio
import random
import time
import urllib.parse
from collections.abc import Callable
from typing import Any

import httpx
import structlog

from connectors.url_security import SSRFValidationError, validate_url_async

logger = structlog.get_logger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
SENSITIVE_HEADER_KEYWORDS = {"auth", "key", "token", "secret", "password", "bearer"}
_REDIRECT_STATUS = {301, 302, 303, 307, 308}
_DROP_RESPONSE_HEADERS = {"content-encoding", "content-length", "transfer-encoding"}


class ResponseTooLargeError(ValueError):
    """Raised when a response body exceeds the configured byte limit."""


class TooManyRedirectsError(ValueError):
    """Raised when the redirect chain exceeds the configured limit."""


class SafeAsyncHttpClient:
    """
    Robust async HTTP client wrapping httpx with exponential backoff and rate pacing.
    """

    def __init__(
        self,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        min_request_interval_seconds: float = 0.5,
        user_agent: str = "AI-Research-Copilot/0.1.0 (https://github.com/ai-research-copilot; research-bot)",
        total_timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.min_request_interval = min_request_interval_seconds
        self.user_agent = user_agent
        self.total_timeout = total_timeout_seconds
        self._transport = transport      # tests only
        self._resolver = resolver        # tests only
        self._last_request_time: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def _sanitize_headers(self, headers: dict[str, Any] | None) -> dict[str, str]:
        """Return a copy of headers with sensitive values masked."""
        if not headers:
            return {}
        safe: dict[str, str] = {}
        for k, v in headers.items():
            k_lower = k.lower()
            if any(s in k_lower for s in SENSITIVE_HEADER_KEYWORDS):
                safe[k] = "[REDACTED]"
            else:
                safe[k] = str(v)
        return safe

    async def _rate_limit_pause(self, host: str) -> None:
        """Enforce minimum delay between calls to the same host."""
        async with self._lock:
            now = time.time()
            last_time = self._last_request_time.get(host, 0.0)
            elapsed = now - last_time
            if elapsed < self.min_request_interval:
                delay = self.min_request_interval - elapsed
                await asyncio.sleep(delay)
            self._last_request_time[host] = time.time()

    # ── Hardened single request (validated redirects + streaming size cap) ────

    async def _hardened_once(
        self,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
        follow_redirects: bool,
        ssrf_protect: bool,
        max_bytes: int | None,
        max_redirects: int,
    ) -> httpx.Response:
        current = url
        timeout_cfg = httpx.Timeout(timeout, connect=min(5.0, timeout))
        redirects = 0
        async with httpx.AsyncClient(
            timeout=timeout_cfg, follow_redirects=False, transport=self._transport, trust_env=False
        ) as client:
            while True:
                req_headers = dict(headers)
                extensions: dict[str, Any] = {}
                target = current
                if ssrf_protect:
                    addrs = await validate_url_async(current, resolver=self._resolver)
                    parsed = urllib.parse.urlparse(current)
                    host = parsed.hostname or ""
                    default_port = 443 if parsed.scheme.lower() == "https" else 80
                    host_header = host if parsed.port in (None, default_port) else f"{host}:{parsed.port}"
                    req_headers["Host"] = host_header
                    if parsed.scheme.lower() == "https":
                        extensions["sni_hostname"] = host
                    target = str(httpx.URL(current).copy_with(host=addrs[0]))

                async with client.stream(
                    "GET",
                    target,
                    params=params if redirects == 0 else None,
                    headers=req_headers,
                    extensions=extensions or None,
                ) as resp:
                    if follow_redirects and resp.status_code in _REDIRECT_STATUS and resp.headers.get("location"):
                        redirects += 1
                        if redirects > max_redirects:
                            raise TooManyRedirectsError(f"Exceeded {max_redirects} redirects.")
                        current = urllib.parse.urljoin(current, resp.headers["location"])
                        continue  # next hop is validated at loop start

                    declared = resp.headers.get("content-length")
                    if max_bytes is not None and declared and declared.isdigit() and int(declared) > max_bytes:
                        raise ResponseTooLargeError(
                            f"Response size ({declared} bytes) exceeds max limit ({max_bytes} bytes)"
                        )
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in resp.aiter_bytes():
                        total += len(chunk)
                        if max_bytes is not None and total > max_bytes:
                            raise ResponseTooLargeError(
                                f"Response size exceeds max limit ({max_bytes} bytes)"
                            )
                        chunks.append(chunk)

                    out_headers = [
                        (k, v) for k, v in resp.headers.multi_items() if k.lower() not in _DROP_RESPONSE_HEADERS
                    ]
                    return httpx.Response(
                        status_code=resp.status_code,
                        headers=out_headers,
                        content=b"".join(chunks),
                        request=httpx.Request("GET", current),
                    )

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        follow_redirects: bool = True,
        ssrf_protect: bool = False,
        max_bytes: int | None = None,
        max_redirects: int = 5,
    ) -> httpx.Response:
        """
        Execute an HTTP GET request with retries, backoff, and rate limit pacing.

        With `ssrf_protect` and/or `max_bytes` the hardened path is used (see module doc).
        SSRFValidationError, ResponseTooLargeError and TooManyRedirectsError are never retried.
        """
        req_headers = {"User-Agent": self.user_agent}
        if headers:
            req_headers.update(headers)

        req_timeout = timeout or self.timeout
        parsed_url = httpx.URL(url)
        host = parsed_url.host
        hardened = ssrf_protect or max_bytes is not None

        attempt = 0
        last_exception: Exception | None = None

        while attempt <= self.max_retries:
            attempt += 1
            await self._rate_limit_pause(host)

            try:
                if hardened:
                    try:
                        async with asyncio.timeout(self.total_timeout):
                            response = await self._hardened_once(
                                url, params, req_headers, req_timeout, follow_redirects,
                                ssrf_protect, max_bytes, max_redirects,
                            )
                    except TimeoutError as te:
                        raise httpx.ReadTimeout("total request deadline exceeded") from te
                else:
                    async with httpx.AsyncClient(
                        timeout=req_timeout,
                        follow_redirects=follow_redirects,
                    ) as client:
                        response = await client.get(url, params=params, headers=req_headers)

                # If successful or client error (not retryable), return immediately
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    return response

                # Handle 429 / 5xx retryable status
                retry_after_str = response.headers.get("Retry-After")
                if retry_after_str and retry_after_str.isdigit():
                    sleep_time = float(retry_after_str)
                else:
                    sleep_time = (self.backoff_factor ** attempt) + random.uniform(0.1, 0.5)

                logger.warning(
                    "http_request_retryable_status",
                    url=str(response.url),
                    status_code=response.status_code,
                    attempt=attempt,
                    retry_after=sleep_time,
                )

                if attempt > self.max_retries:
                    return response

                await asyncio.sleep(sleep_time)

            except (SSRFValidationError, ResponseTooLargeError, TooManyRedirectsError):
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
                last_exception = e
                sleep_time = (self.backoff_factor ** attempt) + random.uniform(0.1, 0.5)
                logger.warning(
                    "http_request_network_error",
                    url=url,
                    error=type(e).__name__,
                    attempt=attempt,
                    retry_after=sleep_time,
                )
                if attempt > self.max_retries:
                    raise last_exception from e
                await asyncio.sleep(sleep_time)

        if last_exception:
            raise last_exception
        raise httpx.RequestError(f"Failed to fetch {url} after {self.max_retries} attempts")

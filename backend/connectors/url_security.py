"""
Outbound URL validation (SSRF defense) shared by the web fetcher and HTTP client.

Checks performed on every URL, including every redirect hop:
  * scheme must be http or https; no embedded credentials; hostname required
  * obvious local hostnames are rejected
  * the hostname is resolved and EVERY returned address must be a public (global)
    address. Alternate numeric spellings (decimal/octal/hex, short forms) are handled
    because validation runs on the resolved address, not the spelling.
  * IPv4-mapped / 6to4 / Teredo IPv6 forms are unwrapped and the embedded IPv4 address
    is validated too.

Limitation: resolve-then-connect is inherently racy (DNS rebinding). The HTTP client
mitigates this by connecting to the validated IP literal (see SafeAsyncHttpClient) but
this module on its own only validates.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
import urllib.parse
from collections.abc import Callable

ALLOWED_SCHEMES = {"http", "https"}
_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain")
_LOCAL_NAMES = {"localhost", "0.0.0.0", "::", "::1", "ip6-localhost"}


class SSRFValidationError(ValueError):
    """Raised when a URL targets a disallowed scheme, host or address."""


Resolver = Callable[[str], list[str]]


def check_ip(ip_str: str) -> None:
    """Raise SSRFValidationError unless `ip_str` is a public unicast address."""
    try:
        ip = ipaddress.ip_address(ip_str.split("%")[0])
    except ValueError as e:
        raise SSRFValidationError(f"Unparseable address '{ip_str}'.") from e

    candidates = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            candidates.append(ip.ipv4_mapped)
        if ip.sixtofour is not None:
            candidates.append(ip.sixtofour)
        if ip.teredo is not None:
            candidates.extend(ip.teredo)

    for c in candidates:
        s = str(c)
        if s == "169.254.169.254":
            raise SSRFValidationError("Access to cloud metadata service is forbidden.")
        if c.is_loopback:
            raise SSRFValidationError(f"Access to loopback address ({s}) is forbidden.")
        if c.is_private:
            raise SSRFValidationError(f"Access to private subnet ({s}) is forbidden.")
        if c.is_link_local:
            raise SSRFValidationError(f"Access to link-local address ({s}) is forbidden.")
        if c.is_reserved or c.is_multicast or c.is_unspecified:
            raise SSRFValidationError(f"Access to special IP address ({s}) is forbidden.")
        if not c.is_global:  # shared address space 100.64/10, benchmarking nets, etc.
            raise SSRFValidationError(f"Access to non-public address ({s}) is forbidden.")


def check_url_shape(url: str) -> urllib.parse.ParseResult:
    """Validate scheme, credentials and hostname syntax (no DNS)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise SSRFValidationError(
            f"Invalid URL scheme '{parsed.scheme}'. Only http and https are allowed."
        )
    if parsed.username or parsed.password:
        raise SSRFValidationError("URLs with embedded credentials are not allowed.")
    host = parsed.hostname
    if not host:
        raise SSRFValidationError("URL contains no valid hostname.")
    h = host.lower().rstrip(".")
    if h in _LOCAL_NAMES or h.endswith(_LOCAL_SUFFIXES) or h == "127.0.0.1":
        raise SSRFValidationError(f"Access to localhost/local hostname '{host}' is forbidden for security.")
    return parsed


def _sync_resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise SSRFValidationError(f"Could not resolve host '{host}': {e}") from e
    return [str(i[4][0]) for i in infos]


async def _async_resolve(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise SSRFValidationError(f"Could not resolve host '{host}': {e}") from e
    return [str(i[4][0]) for i in infos]


def validate_url_sync(url: str, resolver: Resolver | None = None) -> list[str]:
    """Validate URL + resolved addresses. Returns the validated addresses."""
    parsed = check_url_shape(url)
    addrs = (resolver or _sync_resolve)(parsed.hostname or "")
    if not addrs:
        raise SSRFValidationError("Host resolved to no addresses.")
    for a in addrs:
        check_ip(a)
    return addrs


async def validate_url_async(
    url: str, resolver: Callable[[str], list[str]] | None = None
) -> list[str]:
    """Async variant (non-blocking DNS). `resolver` is a sync callable, for tests."""
    parsed = check_url_shape(url)
    host = parsed.hostname or ""
    addrs = resolver(host) if resolver is not None else await _async_resolve(host)
    if not addrs:
        raise SSRFValidationError("Host resolved to no addresses.")
    for a in addrs:
        check_ip(a)
    return addrs

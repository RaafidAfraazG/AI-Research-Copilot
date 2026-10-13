"""
Secret and path redaction for log output and diagnostics.

Two layers:
1. Exact values of configured secrets (API keys, passwords, tokens) are replaced,
   which catches secrets in any surrounding text.
2. Pattern-based redaction of common credential shapes (Google/OpenAI-style keys,
   bearer tokens, key=value pairs, URL userinfo, key= query parameters).

Filesystem paths are normalised rather than erased so tracebacks stay debuggable:
the repository root becomes "<repo>" and the user's home directory becomes "~".
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

REDACTED = "[REDACTED]"

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), REDACTED),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer " + REDACTED),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|passwd|pwd)"
            r"(\s*[=:]\s*)(?!\[REDACTED\])[^\s,;&'\"}]+"
        ),
        r"\1\2" + REDACTED,
    ),
    (re.compile(r"(?i)([?&](?:key|api_key|apikey|token|access_token)=)[^&\s]+"), r"\1" + REDACTED),
    (re.compile(r"(://)[^/\s:@]+:[^@\s/]+@"), r"\1" + REDACTED + "@"),
]

_SECRET_FIELD_HINTS = ("key", "password", "token", "secret")
_MIN_SECRET_LEN = 6


def _configured_secret_values() -> list[str]:
    try:
        from core.config import get_settings

        st = get_settings()
    except Exception:  # settings must never break logging
        return []
    values: list[str] = []
    for name in type(st).model_fields:
        if any(h in name.lower() for h in _SECRET_FIELD_HINTS):
            v = getattr(st, name, None)
            if isinstance(v, str) and len(v) >= _MIN_SECRET_LEN:
                values.append(v)
    # longest first so a secret containing another is replaced whole
    return sorted(set(values), key=len, reverse=True)


def _normalise_paths(text: str) -> str:
    repo = str(Path(__file__).resolve().parent.parent.parent)
    home = str(Path.home())
    for raw, repl in ((repo, "<repo>"), (home, "~")):
        if raw:
            text = text.replace(raw, repl).replace(raw.replace("\\", "/"), repl)
    return text


def redact(text: str) -> str:
    """Return `text` with secrets redacted and local paths normalised."""
    if not text:
        return text
    for secret in _configured_secret_values():
        if secret in text:
            text = text.replace(secret, REDACTED)
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return _normalise_paths(text)


def redact_value(value: Any, _depth: int = 0) -> Any:
    """Redact strings inside (shallow-nested) dict/list/tuple structures."""
    if isinstance(value, str):
        return redact(value)
    if _depth >= 3:
        return value
    if isinstance(value, dict):
        return {k: redact_value(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact_value(v, _depth + 1) for v in value)
    return value


def redacting_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor: redact every logged value and any attached traceback."""
    exc_info = event_dict.pop("exc_info", None)
    if exc_info:
        import traceback

        if exc_info is True:
            import sys

            exc_info = sys.exc_info()
        if isinstance(exc_info, BaseException):
            exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
        if isinstance(exc_info, tuple) and exc_info[0] is not None:
            event_dict["exception"] = redact("".join(traceback.format_exception(*exc_info)))
    return {k: redact_value(v) for k, v in event_dict.items()}

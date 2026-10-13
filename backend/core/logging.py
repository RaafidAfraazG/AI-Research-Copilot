"""
Structured logging configuration using structlog.

Design:
- Development: colourised human-readable console output via ConsoleRenderer.
- Production: JSON-structured output for log aggregators (Datadog, Loki, etc.).
- Integrates with stdlib logging so uvicorn/SQLAlchemy logs are also captured.
- Noisy libraries are quieted to WARNING level.
- Never log secrets — this is enforced by convention; see the note below.

Security note:
    Do NOT pass API keys, passwords, or auth tokens as structlog bound variables.
    Use `.bind(user_id=..., request_id=...)` for safe context fields only.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from core.redaction import redacting_processor


def setup_logging(log_level: str = "INFO", is_development: bool = True) -> None:
    """
    Configure structlog with stdlib integration.

    Call this once at application startup before any logging occurs.
    It is safe to call multiple times (re-entrant).

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
        is_development: When True, use pretty console output; otherwise JSON.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Processors applied to every log event regardless of output format
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,      # Merge context from contextvars
        structlog.stdlib.add_logger_name,             # Add 'logger' field
        structlog.stdlib.add_log_level,               # Add 'level' field
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        redacting_processor,  # redact secrets/paths in every event and traceback
    ]

    if is_development:
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    else:
        # Production: structured JSON that log aggregators can parse
        renderer = structlog.processors.JSONRenderer()

    # Configure ProcessorFormatter to handle both structlog and stdlib log records
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Replace all existing handlers on the root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Quiet down noisy third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("opensearch").setLevel(logging.WARNING)

    # Configure structlog to use the same stdlib handler pipeline
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Convenience wrapper — returns a structlog logger bound to a name."""
    return structlog.get_logger(name)

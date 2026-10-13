"""
Application exception hierarchy.

All custom exceptions inherit from AppException so that the global
exception handler in main.py can intercept them uniformly.

Usage:
    raise NotFoundError("Research session not found")
    raise InfrastructureError("PostgreSQL unreachable")
"""
from __future__ import annotations


class AppException(Exception):  # noqa: N818 — deliberate: base class, not a raised exception directly
    """Base class for all application-level exceptions."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class NotFoundError(AppException):
    """Raised when a requested resource does not exist."""
    status_code = 404
    code = "not_found"


class ConflictError(AppException):
    """Raised when an operation conflicts with existing state."""
    status_code = 409
    code = "conflict"


class ValidationError(AppException):
    """Raised when request data fails business-logic validation."""
    status_code = 422
    code = "validation_error"


class InfrastructureError(AppException):
    """Raised when a required infrastructure service is unavailable."""
    status_code = 503
    code = "infrastructure_unavailable"


class ConfigurationError(AppException):
    """Raised when application configuration is invalid or missing."""
    status_code = 500
    code = "configuration_error"


class ExternalServiceError(AppException):
    """Raised when an external API (arXiv, Semantic Scholar, etc.) returns an error."""
    status_code = 502
    code = "external_service_error"

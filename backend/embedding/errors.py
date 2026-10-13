"""Embedding-layer errors. Messages are safe to return to API clients."""
from __future__ import annotations

from core.exceptions import AppException


class EmbeddingError(AppException):
    status_code = 503
    code = "embedding_unavailable"


class EmbeddingModelLoadError(EmbeddingError):
    """The configured embedding model could not be loaded. No fallback is attempted."""
    code = "embedding_model_load_failed"


class EmbeddingConfigurationError(EmbeddingError):
    status_code = 500
    code = "embedding_configuration_error"

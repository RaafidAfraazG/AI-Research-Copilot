"""
Factory for creating the EmbeddingProvider selected by configuration.

Fail-closed policy: the mock provider is returned ONLY when
EMBEDDING_PROVIDER=mock is configured explicitly. A missing/broken real model
raises EmbeddingModelLoadError (lazily, on first use) instead of silently
substituting mock vectors.
"""
from __future__ import annotations

from functools import lru_cache

from core.config import Settings, get_settings
from embedding.base import EmbeddingProvider
from embedding.errors import EmbeddingConfigurationError
from embedding.mock_provider import MockEmbeddingProvider
from embedding.openai_provider import OpenAIEmbeddingProvider
from embedding.sentence_transformer_provider import SentenceTransformerProvider


def create_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Construct the configured provider (model loading itself is lazy and raises on failure)."""
    if settings is None:
        settings = get_settings()

    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set when EMBEDDING_PROVIDER is 'openai'")
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model or "text-embedding-3-small",
        )
    if settings.embedding_provider == "local":
        return SentenceTransformerProvider(
            model_name=settings.embedding_model,
            normalize_embeddings=settings.embedding_normalize,
        )
    if settings.embedding_provider == "mock":
        return MockEmbeddingProvider(dimension=384)
    raise EmbeddingConfigurationError(
        f"Unknown embedding provider '{settings.embedding_provider}'."
    )


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Cached singleton of the configured embedding provider."""
    return create_embedding_provider()

"""Embedding identity: what exactly produced a set of vectors."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from embedding.mock_provider import MockEmbeddingProvider


class EmbeddingIdentity(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    provider: str                 # "sentence_transformers" | "openai" | "mock" | other
    model_name: str
    dimension: int
    revision: str | None = None   # model commit hash when discoverable
    normalized: bool = True
    is_mock: bool = False


def describe_embedding_provider(provider: Any) -> EmbeddingIdentity:
    """
    Build an identity from a live provider. For lazily-loading providers this
    loads the model (and may raise EmbeddingModelLoadError) — intentionally:
    identity must come from the real loaded model, not from configuration.
    """
    from embedding.sentence_transformer_provider import SentenceTransformerProvider

    is_mock = isinstance(provider, MockEmbeddingProvider)
    if is_mock:
        kind = "mock"
    elif isinstance(provider, SentenceTransformerProvider):
        kind = "sentence_transformers"
    else:
        kind = type(provider).__name__.replace("EmbeddingProvider", "").lower() or "unknown"

    return EmbeddingIdentity(
        provider=kind,
        model_name=str(provider.model_name),
        dimension=int(provider.dimension),
        revision=getattr(provider, "revision", None),
        normalized=bool(getattr(provider, "normalize_embeddings", True)),
        is_mock=is_mock,
    )

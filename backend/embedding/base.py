"""
Embedding provider interface protocol and common data structures.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol defining the interface for all text embedding providers.
    
    Implementations may be local (e.g. SentenceTransformers), remote API (e.g. OpenAI),
    or mock (for testing).
    """

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality produced by this embedding provider."""
        ...

    @property
    def model_name(self) -> str:
        """Return the identifier/name of the model used for embedding."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate dense vector embeddings for a list of document chunks or texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vectors, each of length self.dimension.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """
        Generate a single dense vector embedding for a search query.

        Args:
            text: Query string to embed.

        Returns:
            Float vector of length self.dimension.
        """
        ...

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed_texts."""
        ...

    async def aembed_query(self, text: str) -> list[float]:
        """Async version of embed_query."""
        ...

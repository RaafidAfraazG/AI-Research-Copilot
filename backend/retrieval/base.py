"""
Base interface for all search retrievers (lexical, dense, hybrid).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from retrieval.models import RetrievalQuery, RetrievalResult


class BaseRetriever(ABC):
    """
    Abstract base class for document retrievers.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the retriever (e.g. 'bm25', 'dense', 'hybrid')."""
        ...

    @abstractmethod
    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Execute search for candidate document passages.

        Args:
            query: RetrievalQuery containing search parameters.

        Returns:
            RetrievalResult containing scored chunks and execution metadata.
        """
        ...

    @abstractmethod
    async def ping(self) -> bool:
        """
        Health check to determine if the backing search service is reachable.
        """
        ...

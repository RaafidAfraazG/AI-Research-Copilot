"""
Abstract Base Class for Research Source Connectors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from connectors.models import (
    ConnectorSearchRequest,
    ConnectorSearchResponse,
    ResearchDocument,
)


class ResearchSourceConnector(ABC):
    """
    Abstract base class for external research data source connectors
    (e.g., arXiv, Semantic Scholar, Web Fetcher).
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Identifier for this data source (e.g. 'arxiv', 'semantic_scholar', 'web')."""
        ...

    @abstractmethod
    async def search(self, request: ConnectorSearchRequest) -> ConnectorSearchResponse:
        """
        Search the external source and return normalized ResearchDocument instances.
        """
        ...

    @abstractmethod
    async def fetch(self, identifier: str) -> ResearchDocument | None:
        """
        Fetch a single document by its canonical source identifier or URL.
        """
        ...

    @abstractmethod
    async def health(self) -> tuple[bool, str]:
        """
        Check if the external data source API is reachable.
        Returns (is_healthy, status_message).
        """
        ...

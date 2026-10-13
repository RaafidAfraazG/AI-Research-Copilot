"""
Connector Ingestion Service.
Coordinates fetching documents from external connectors, deduplication, and
indexing into the retrieval system (OpenSearch + Qdrant + PostgreSQL).
"""
from __future__ import annotations

import time
from typing import Any
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from connectors.arxiv_connector import ArxivConnector
from connectors.base import ResearchSourceConnector
from connectors.deduplication import DocumentDeduplicator
from connectors.models import (
    ConnectorSearchRequest,
    ResearchDocument,
)
from connectors.semantic_scholar_connector import SemanticScholarConnector
from connectors.web_fetcher import WebFetcher
from retrieval.indexing_service import IndexingService
from retrieval.models import IngestDocumentResponse

logger = structlog.get_logger(__name__)


class ConnectorIngestionService:
    """
    End-to-end ingestion pipeline:
    Connector -> ResearchDocument -> Deduplication -> PassageChunking -> Embedding -> Dual-Index.
    """

    def __init__(
        self,
        indexing_service: IndexingService | None = None,
        deduplicator: DocumentDeduplicator | None = None,
    ) -> None:
        self.indexing_service = indexing_service or IndexingService()
        self.deduplicator = deduplicator or DocumentDeduplicator()
        
        # Initialize standard connectors
        self.connectors: dict[str, ResearchSourceConnector] = {
            "arxiv": ArxivConnector(),
            "semantic_scholar": SemanticScholarConnector(),
            "web": WebFetcher(),
        }

    def get_connector(self, source_type: str) -> ResearchSourceConnector:
        """Get connector instance by source name."""
        connector = self.connectors.get(source_type.lower())
        if not connector:
            raise ValueError(f"Unknown source connector '{source_type}'. Available: {list(self.connectors.keys())}")
        return connector

    async def ingest_document(
        self,
        doc: ResearchDocument,
        db_session: AsyncSession | None = None,
    ) -> IngestDocumentResponse:
        """
        Deduplicate, validate, and index a single ResearchDocument into the retrieval system.
        """
        # 1. Deduplicate & Merge
        clean_doc, is_new = self.deduplicator.register_and_merge(doc)

        # 2. Convert to Indexing Request
        ingest_req = clean_doc.to_ingest_request()

        # 3. Index via IndexingService
        return await self.indexing_service.index_document(ingest_req, db_session=db_session)

    async def search_and_ingest(
        self,
        source_type: str,
        query: str,
        limit: int = 5,
        db_session: AsyncSession | None = None,
    ) -> list[IngestDocumentResponse]:
        """
        Search an external source connector, normalize results, deduplicate, and index.
        """
        connector = self.get_connector(source_type)
        search_req = ConnectorSearchRequest(query=query, limit=limit)
        search_resp = await connector.search(search_req)

        results: list[IngestDocumentResponse] = []
        for doc in search_resp.documents:
            res = await self.ingest_document(doc, db_session=db_session)
            results.append(res)

        return results

    async def fetch_and_ingest(
        self,
        source_type: str,
        identifier_or_url: str,
        db_session: AsyncSession | None = None,
    ) -> IngestDocumentResponse | None:
        """
        Fetch a specific document from a source and index it.
        """
        connector = self.get_connector(source_type)
        doc = await connector.fetch(identifier_or_url)
        if not doc:
            return None
        return await self.ingest_document(doc, db_session=db_session)

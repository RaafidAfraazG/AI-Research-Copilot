"""Research source connectors package."""
from connectors.arxiv_connector import ArxivConnector
from connectors.base import ResearchSourceConnector
from connectors.deduplication import DocumentDeduplicator, extract_canonical_id, normalize_url
from connectors.http_client import SafeAsyncHttpClient
from connectors.ingestion_service import ConnectorIngestionService
from connectors.models import (
    CitationInfo,
    ConnectorSearchRequest,
    ConnectorSearchResponse,
    ProvenanceInfo,
    ResearchDocument,
)
from connectors.semantic_scholar_connector import SemanticScholarConnector
from connectors.web_fetcher import SSRFValidationError, WebFetcher

__all__ = [
    "ResearchSourceConnector",
    "ArxivConnector",
    "SemanticScholarConnector",
    "WebFetcher",
    "SafeAsyncHttpClient",
    "DocumentDeduplicator",
    "ConnectorIngestionService",
    "ResearchDocument",
    "CitationInfo",
    "ProvenanceInfo",
    "ConnectorSearchRequest",
    "ConnectorSearchResponse",
    "SSRFValidationError",
    "normalize_url",
    "extract_canonical_id",
]

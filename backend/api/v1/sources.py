"""
API endpoints for searching external sources and ingesting research documents.
"""
from __future__ import annotations

from typing import Any, Literal
from api.deps import get_indexing_service
from fastapi import APIRouter, Depends, HTTPException, Query, status
import structlog
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from connectors.arxiv_connector import ArxivConnector
from connectors.ingestion_service import ConnectorIngestionService
from connectors.models import (
    ConnectorSearchRequest,
    ConnectorSearchResponse,
    ResearchDocument,
)
from connectors.semantic_scholar_connector import SemanticScholarConnector
from connectors.http_client import ResponseTooLargeError, TooManyRedirectsError
from connectors.url_security import SSRFValidationError
from connectors.web_fetcher import WebFetcher
from db.database import get_db_session
from retrieval.models import IngestDocumentResponse

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/sources", tags=["sources"])


class IngestSourceRequest(BaseModel):
    """Request model for fetching from a source and ingesting into search index."""
    source_type: Literal["arxiv", "semantic_scholar", "web"] = Field(
        ...,
        description="Source connector to use",
    )
    query_or_identifier: str = Field(
        ...,
        min_length=1,
        description="Search query (e.g. 'RAG'), paper identifier (e.g. '2005.11401'), or URL",
    )
    limit: int = Field(default=5, ge=1, le=20, description="Max papers to fetch and index")


class IngestSourceResponse(BaseModel):
    """Response model summarizing ingestion across source documents."""
    source_type: str
    total_ingested: int
    results: list[IngestDocumentResponse]


@router.get(
    "/arxiv/search",
    response_model=ConnectorSearchResponse,
    summary="Search arXiv papers",
    description="Search arXiv by keyword and return normalized ResearchDocument instances.",
)
async def search_arxiv(
    query: str = Query(..., min_length=1, description="Keywords to search"),
    limit: int = Query(default=10, ge=1, le=50, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Result offset"),
) -> ConnectorSearchResponse:
    """Search arXiv repository."""
    connector = ArxivConnector()
    req = ConnectorSearchRequest(query=query, limit=limit, offset=offset)
    return await connector.search(req)


@router.get(
    "/semantic-scholar/search",
    response_model=ConnectorSearchResponse,
    summary="Search Semantic Scholar",
    description="Search papers and citation graphs via Semantic Scholar Academic Graph API.",
)
async def search_semantic_scholar(
    query: str = Query(..., min_length=1, description="Keywords to search"),
    limit: int = Query(default=10, ge=1, le=50, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Result offset"),
    year: int | None = Query(default=None, ge=1900, le=2050, description="Filter publication year"),
) -> ConnectorSearchResponse:
    """Search Semantic Scholar database."""
    connector = SemanticScholarConnector()
    req = ConnectorSearchRequest(query=query, limit=limit, offset=offset, year=year)
    return await connector.search(req)


@router.get(
    "/web/fetch",
    response_model=ResearchDocument,
    summary="Fetch and extract web document",
    description="Safely fetch a specific URL, validate SSRF security, strip boilerplate, and return document.",
)
async def fetch_web_page(
    url: str = Query(..., min_length=1, description="URL of document to fetch"),
) -> ResearchDocument:
    """Safely fetch and parse a web page."""
    fetcher = WebFetcher()
    try:
        doc = await fetcher.fetch(url)
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Could not extract content from URL")
        return doc
    except SSRFValidationError as e:
        log.warning("web_fetch_blocked", reason=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Security check failed: the URL uses an unsupported scheme or a disallowed destination.",
        ) from e
    except (ResponseTooLargeError, TooManyRedirectsError, ValueError) as e:
        log.info("web_fetch_rejected", error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The remote document was rejected (too large, too many redirects, or an unsupported type).",
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        log.warning("web_fetch_failed", error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch the external document.",
        ) from e


@router.post(
    "/ingest",
    response_model=IngestSourceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest from source into retrieval index",
    description="Fetch papers from arXiv, Semantic Scholar, or Web and index into OpenSearch and Qdrant.",
)
async def ingest_from_source(
    request: IngestSourceRequest,
    db: AsyncSession = Depends(get_db_session),
) -> IngestSourceResponse:
    """Fetch external documents and index them into the retrieval system."""
    ingestion_service = ConnectorIngestionService(indexing_service=get_indexing_service())
    
    try:
        if request.source_type == "web":
            res = await ingestion_service.fetch_and_ingest(
                source_type="web",
                identifier_or_url=request.query_or_identifier,
                db_session=db,
            )
            results = [res] if res else []
        elif "/" in request.query_or_identifier or "." in request.query_or_identifier and len(request.query_or_identifier) < 20:
            # Identifier fetch (e.g. arXiv ID '2005.11401')
            res = await ingestion_service.fetch_and_ingest(
                source_type=request.source_type,
                identifier_or_url=request.query_or_identifier,
                db_session=db,
            )
            results = [res] if res else []
        else:
            # Query search and ingest
            results = await ingestion_service.search_and_ingest(
                source_type=request.source_type,
                query=request.query_or_identifier,
                limit=request.limit,
                db_session=db,
            )

        return IngestSourceResponse(
            source_type=request.source_type,
            total_ingested=len(results),
            results=results,
        )
    except SSRFValidationError as e:
        log.warning("ingest_blocked", reason=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Security check failed: the URL uses an unsupported scheme or a disallowed destination.",
        ) from e

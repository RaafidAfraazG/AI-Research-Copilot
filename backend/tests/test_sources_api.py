"""
Unit tests for Sources API endpoints (backend/api/v1/sources.py).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from connectors.models import (
    CitationInfo,
    ConnectorSearchResponse,
    ProvenanceInfo,
    ResearchDocument,
)
from connectors.web_fetcher import SSRFValidationError
from main import app
from retrieval.models import IngestDocumentResponse


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def sample_research_doc() -> ResearchDocument:
    return ResearchDocument(
        source_type="arxiv",
        source_id="2005.11401",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis", "Ethan Perez"],
        abstract="We explore RAG architecture...",
        url="https://arxiv.org/abs/2005.11401",
        citation_info=CitationInfo(arxiv_id="2005.11401"),
        provenance=ProvenanceInfo(
            source_type="arxiv",
            source_id="2005.11401",
            canonical_url="https://arxiv.org/abs/2005.11401",
        ),
    )


@pytest.mark.asyncio
class TestSourcesApiEndpoints:
    @patch("connectors.arxiv_connector.ArxivConnector.search")
    async def test_arxiv_search_endpoint(self, mock_search: AsyncMock, sample_research_doc: ResearchDocument) -> None:
        mock_search.return_value = ConnectorSearchResponse(
            source_type="arxiv",
            query="retrieval augmented generation",
            total_results=1,
            documents=[sample_research_doc],
        )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/api/v1/sources/arxiv/search?query=retrieval+augmented+generation&limit=5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["source_type"] == "arxiv"
            assert data["total_results"] == 1
            assert data["documents"][0]["source_id"] == "2005.11401"
            assert "Retrieval-Augmented" in data["documents"][0]["title"]

    @patch("connectors.semantic_scholar_connector.SemanticScholarConnector.search")
    async def test_semantic_scholar_search_endpoint(self, mock_search: AsyncMock, sample_research_doc: ResearchDocument) -> None:
        s2_doc = sample_research_doc.model_copy(update={"source_type": "semantic_scholar"})
        mock_search.return_value = ConnectorSearchResponse(
            source_type="semantic_scholar",
            query="dense retrieval",
            total_results=1,
            documents=[s2_doc],
        )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/api/v1/sources/semantic-scholar/search?query=dense+retrieval&limit=5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["source_type"] == "semantic_scholar"
            assert len(data["documents"]) == 1

    @patch("connectors.web_fetcher.WebFetcher.fetch")
    async def test_web_fetch_endpoint_success(self, mock_fetch: AsyncMock) -> None:
        web_doc = ResearchDocument(
            source_type="web",
            source_id="https://example.com/ai-post",
            title="AI Post Title",
            abstract="Summary of AI post",
            full_text="Full text of the post.",
            url="https://example.com/ai-post",
        )
        mock_fetch.return_value = web_doc

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/api/v1/sources/web/fetch?url=https://example.com/ai-post")
            assert resp.status_code == 200
            data = resp.json()
            assert data["source_type"] == "web"
            assert data["title"] == "AI Post Title"

    @patch("connectors.web_fetcher.WebFetcher.fetch")
    async def test_web_fetch_ssrf_blocked(self, mock_fetch: AsyncMock) -> None:
        mock_fetch.side_effect = SSRFValidationError("Access to localhost is forbidden.")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/api/v1/sources/web/fetch?url=http://localhost:8080/secret")
            assert resp.status_code == 400
            assert "Security check failed" in resp.json()["detail"]

    @patch("connectors.ingestion_service.ConnectorIngestionService.search_and_ingest")
    async def test_ingest_source_endpoint(self, mock_search_ingest: AsyncMock) -> None:
        mock_search_ingest.return_value = [
            IngestDocumentResponse(
                document_id="doc-123",
                source_type="arxiv",
                source_id="2005.11401",
                title="RAG Paper",
                total_chunks=3,
                indexed_in_opensearch=True,
                indexed_in_qdrant=True,
                status="success",
            )
        ]

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v1/sources/ingest",
                json={
                    "source_type": "arxiv",
                    "query_or_identifier": "retrieval augmented generation",
                    "limit": 2,
                },
            )
            assert resp.status_code == 201
            data = resp.json()
            assert data["source_type"] == "arxiv"
            assert data["total_ingested"] == 1
            assert data["results"][0]["document_id"] == "doc-123"
            assert data["results"][0]["total_chunks"] == 3

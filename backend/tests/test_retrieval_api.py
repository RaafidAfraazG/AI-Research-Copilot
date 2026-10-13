"""
Unit & integration tests for Retrieval API endpoints (api/v1/retrieval.py).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from main import app
from retrieval.models import RetrievalResult, ScoredChunk


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
class TestRetrievalApiEndpoints:
    async def test_chunk_preview_endpoint(self) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/retrieval/chunk",
                json={
                    "text": "First sentence is clear. Second sentence has details.",
                    "chunk_size_tokens": 100,
                    "chunk_overlap_tokens": 20,
                    "title": "Preview Test",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["total_chunks"] >= 1
            assert len(data["chunks"]) >= 1
            assert data["chunks"][0]["title"] == "Preview Test"

    async def test_chunk_preview_invalid_overlap(self) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/retrieval/chunk",
                json={
                    "text": "Some text",
                    "chunk_size_tokens": 50,
                    "chunk_overlap_tokens": 50,  # Invalid: overlap >= size
                },
            )
            assert response.status_code == 422

    @patch("api.v1.retrieval.HybridRetriever.search")
    async def test_search_endpoint_hybrid_mode(self, mock_hybrid_search: AsyncMock) -> None:
        mock_hybrid_search.return_value = RetrievalResult(
            query_text="RAG systems",
            retriever="hybrid",
            chunks=[
                ScoredChunk(
                    chunk_id="chk-1",
                    document_id="doc-1",
                    source_type="arxiv",
                    source_id="2005.11401",
                    title="RAG Paper",
                    content="RAG combines parametric memory.",
                    score=0.032,
                    retriever_name="hybrid",
                    rank=1,
                    bm25_rank=1,
                    dense_rank=1,
                )
            ],
            total_found=1,
            latency_ms=15.4,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/retrieval/search",
                json={
                    "query": "RAG systems",
                    "top_k": 5,
                    "mode": "hybrid",
                    "alpha": 0.5,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["query_text"] == "RAG systems"
            assert data["retriever"] == "hybrid"
            assert len(data["chunks"]) == 1
            assert data["chunks"][0]["chunk_id"] == "chk-1"
            assert data["chunks"][0]["bm25_rank"] == 1
            assert data["chunks"][0]["dense_rank"] == 1

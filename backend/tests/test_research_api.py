"""
Unit tests for Research API endpoints (/api/v1/research/retrieve).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from main import app
from ranking.evidence import EvidencePack, EvidencePassage
from retrieval.models import ScoredChunk
from services.research_retrieval_service import (
    LatencyBreakdown,
    ResearchRetrievalResponse,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_research_retrieve_endpoint():
    mock_resp = ResearchRetrievalResponse(
        query="RAG systems",
        total_candidates=1,
        total_evidence=1,
        candidates=[
            ScoredChunk(
                chunk_id="c1",
                document_id="d1",
                source_type="arxiv",
                source_id="2005.11401",
                title="RAG Paper",
                content="RAG combines parametric memory.",
                score=0.03,
                rank=1,
            )
        ],
        reranked=[
            ScoredChunk(
                chunk_id="c1",
                document_id="d1",
                source_type="arxiv",
                source_id="2005.11401",
                title="RAG Paper",
                content="RAG combines parametric memory.",
                rerank_score=8.5,
                score=8.5,
                rank=1,
            )
        ],
        evidence_pack=EvidencePack(
            query="RAG systems",
            selected_passages=[
                EvidencePassage(
                    passage_id="c1",
                    document_id="d1",
                    text="RAG combines parametric memory.",
                    title="RAG Paper",
                    source_type="arxiv",
                    source_id="2005.11401",
                    relevance_score=8.5,
                    rank=1,
                )
            ],
            total_candidates=1,
            total_evidence=1,
        ),
        latency_breakdown=LatencyBreakdown(
            bm25_latency_ms=1.0,
            dense_latency_ms=2.0,
            hybrid_fusion_latency_ms=0.5,
            rerank_latency_ms=10.0,
            evidence_selection_latency_ms=0.5,
            total_retrieval_latency_ms=14.0,
        ),
    )

    with patch("services.research_retrieval_service.ResearchRetrievalService.retrieve_evidence", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = mock_resp

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v1/research/retrieve",
                json={
                    "query": "RAG systems",
                    "candidate_k": 10,
                    "evidence_k": 3,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["query"] == "RAG systems"
            assert data["total_candidates"] == 1
            assert data["total_evidence"] == 1
            assert len(data["evidence_pack"]["selected_passages"]) == 1
            assert data["evidence_pack"]["selected_passages"][0]["relevance_score"] == 8.5
            assert data["latency_breakdown"]["rerank_latency_ms"] == 10.0

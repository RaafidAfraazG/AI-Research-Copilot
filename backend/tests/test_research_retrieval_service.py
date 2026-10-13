"""
Unit tests for ResearchRetrievalService orchestrating hybrid retrieval, reranking, and evidence packaging.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from ranking.evidence import EvidencePack, EvidencePassage
from ranking.evidence_selector import EvidenceSelector
from ranking.ranking_service import RankingService
from retrieval.models import RetrievalResult, ScoredChunk
from services.research_retrieval_service import (
    ResearchRetrievalRequest,
    ResearchRetrievalService,
)


@pytest.mark.asyncio
async def test_research_retrieval_service_orchestration():
    mock_retriever = AsyncMock()
    mock_ranking = AsyncMock()
    mock_selector = MagicMock()

    candidate_chunks = [
        ScoredChunk(
            chunk_id="c1",
            document_id="d1",
            source_type="arxiv",
            source_id="2005.11401",
            title="RAG Paper",
            content="RAG combines dense retrieval and generation.",
            score=0.03,
            rank=1,
        )
    ]
    mock_retriever.search.return_value = RetrievalResult(
        query_text="RAG",
        retriever="hybrid",
        chunks=candidate_chunks,
        total_found=1,
        metadata={"bm25_latency_ms": 1.2, "dense_latency_ms": 2.3, "fusion_latency_ms": 0.4},
    )

    reranked_chunks = [
        ScoredChunk(
            chunk_id="c1",
            document_id="d1",
            source_type="arxiv",
            source_id="2005.11401",
            title="RAG Paper",
            content="RAG combines dense retrieval and generation.",
            rerank_score=8.9,
            score=8.9,
            rank=1,
            retriever_name="cross_encoder",
        )
    ]
    mock_ranking.rerank_candidates.return_value = (reranked_chunks, 12.5)
    mock_ranking.reranker.model_name = "ms-marco-MiniLM"
    mock_ranking.reranker.device = "cpu"

    evidence_pack = EvidencePack(
        query="RAG",
        selected_passages=[
            EvidencePassage(
                passage_id="c1",
                document_id="d1",
                text="RAG combines dense retrieval and generation.",
                title="RAG Paper",
                source_type="arxiv",
                source_id="2005.11401",
                relevance_score=8.9,
                rank=1,
            )
        ],
        total_candidates=1,
        total_evidence=1,
    )
    mock_selector.select.return_value = (evidence_pack, 0.8)

    service = ResearchRetrievalService(
        hybrid_retriever=mock_retriever,
        ranking_service=mock_ranking,
        evidence_selector=mock_selector,
    )

    req = ResearchRetrievalRequest(query="RAG", candidate_k=10, evidence_k=3)
    response = await service.retrieve_evidence(req)

    assert response.query == "RAG"
    assert response.total_candidates == 1
    assert response.total_evidence == 1
    assert len(response.candidates) == 1
    assert len(response.reranked) == 1
    assert response.evidence_pack.selected_passages[0].passage_id == "c1"

    lb = response.latency_breakdown
    assert lb.bm25_latency_ms == 1.2
    assert lb.dense_latency_ms == 2.3
    assert lb.rerank_latency_ms == 12.5
    assert lb.total_retrieval_latency_ms >= 0.0

"""
Unit tests for CrossEncoderReranker and RankingService.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from ranking.base import RerankingError
from ranking.cross_encoder_reranker import CrossEncoderReranker
from ranking.ranking_service import RankingService
from retrieval.models import ScoredChunk


@pytest.fixture
def sample_candidates() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk_id="chk-1",
            document_id="doc-1",
            source_type="arxiv",
            source_id="2005.11401",
            title="RAG Paper",
            content="Retrieval-augmented generation models combine neural parametric and non-parametric memory.",
            score=0.032,
            rank=1,
            retriever_name="hybrid",
        ),
        ScoredChunk(
            chunk_id="chk-2",
            document_id="doc-2",
            source_type="arxiv",
            source_id="1706.03762",
            title="Transformer Paper",
            content="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
            score=0.028,
            rank=2,
            retriever_name="hybrid",
        ),
        ScoredChunk(
            chunk_id="chk-3",
            document_id="doc-3",
            source_type="arxiv",
            source_id="2004.04906",
            title="DPR Paper",
            content="Dense passage retrieval uses dual encoders for fast dense indexing in open-domain QA.",
            score=0.025,
            rank=3,
            retriever_name="hybrid",
        ),
    ]


@pytest.mark.asyncio
async def test_reranker_with_mocked_model(sample_candidates: list[ScoredChunk]):
    """Verify reranker properly scores, orders, and assigns new ranks."""
    reranker = CrossEncoderReranker(model_name="test-model", device="cpu")

    # Mock the score_pairs method to return specific cross-encoder scores
    # chk-3 gets highest score (8.5), chk-1 gets (7.2), chk-2 gets (1.1)
    reranker.score_pairs = MagicMock(return_value=[7.2, 1.1, 8.5])  # type: ignore

    reranked = await reranker.rerank(
        query="How does dense passage retrieval work?",
        candidates=sample_candidates,
        top_k=2,
    )

    assert len(reranked) == 2
    # First should be chk-3 (score 8.5)
    assert reranked[0].chunk_id == "chk-3"
    assert reranked[0].rank == 1
    assert reranked[0].rerank_score == 8.5
    assert reranked[0].retriever_name == "cross_encoder"
    assert reranked[0].metadata["original_rank"] == 3

    # Second should be chk-1 (score 7.2)
    assert reranked[1].chunk_id == "chk-1"
    assert reranked[1].rank == 2
    assert reranked[1].rerank_score == 7.2


@pytest.mark.asyncio
async def test_reranker_empty_candidates():
    """Verify empty candidate list returns empty list immediately."""
    reranker = CrossEncoderReranker()
    reranked = await reranker.rerank(query="Test Query", candidates=[])
    assert reranked == []


@pytest.mark.asyncio
async def test_reranker_error_mode():
    """Verify error is raised when fallback_mode is 'error' and model cannot load."""
    reranker = CrossEncoderReranker(model_name="nonexistent-model", fallback_mode="error")
    reranker._load_model = MagicMock(side_effect=Exception("Failed to load"))  # type: ignore

    candidate = ScoredChunk(
        chunk_id="c1",
        document_id="d1",
        source_type="web",
        source_id="url",
        content="Some content",
    )

    with pytest.raises(Exception):
        await reranker.rerank(query="Test Query", candidates=[candidate])


@pytest.mark.asyncio
async def test_ranking_service_measures_latency(sample_candidates: list[ScoredChunk]):
    """Verify RankingService coordinates reranking and returns latency."""
    reranker = CrossEncoderReranker()
    reranker.score_pairs = MagicMock(return_value=[5.0, 4.0, 3.0])  # type: ignore

    service = RankingService(reranker=reranker)
    reranked, latency = await service.rerank_candidates("RAG", sample_candidates)

    assert len(reranked) == 3
    assert latency >= 0.0
    assert reranked[0].chunk_id == "chk-1"

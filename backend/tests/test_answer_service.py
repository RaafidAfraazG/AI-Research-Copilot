"""
Unit tests for ResearchAnswerService end-to-end synthesis pipeline.
"""
from __future__ import annotations

from unittest.mock import AsyncMock
import pytest

from llm.mock_provider import MockLLMProvider
from ranking.evidence import EvidencePack, EvidencePassage
from services.research_retrieval_service import (
    LatencyBreakdown,
    ResearchRetrievalResponse,
)
from synthesis.answer_service import (
    ResearchAnswerRequest,
    ResearchAnswerService,
)
from synthesis.models import SupportStatus


@pytest.fixture
def sample_retrieval_response():
    passages = [
        EvidencePassage(
            passage_id="p-01",
            document_id="doc-01",
            text="Retrieval-augmented generation models combine neural parametric and non-parametric memory.",
            title="RAG Paper",
            source_type="arxiv",
            source_id="2005.11401",
            url="https://arxiv.org/abs/2005.11401",
            relevance_score=9.1,
            rank=1,
            metadata={"authors": ["Patrick Lewis", "Ethan Perez"], "year": 2020},
        )
    ]
    pack = EvidencePack(
        query="What is RAG?",
        selected_passages=passages,
        total_candidates=10,
        total_evidence=1,
    )
    return ResearchRetrievalResponse(
        query="What is RAG?",
        total_candidates=10,
        total_evidence=1,
        candidates=[],
        reranked=[],
        evidence_pack=pack,
        latency_breakdown=LatencyBreakdown(total_retrieval_latency_ms=15.0),
    )


@pytest.mark.asyncio
async def test_research_answer_service_success(sample_retrieval_response):
    mock_retrieval_svc = AsyncMock()
    mock_retrieval_svc.retrieve_evidence.return_value = sample_retrieval_response

    mock_llm = MockLLMProvider()
    service = ResearchAnswerService(
        retrieval_service=mock_retrieval_svc,
        llm_provider=mock_llm,
    )

    req = ResearchAnswerRequest(query="What is RAG?", candidate_k=10, evidence_k=3)
    answer = await service.answer_question(req)

    assert answer.question == "What is RAG?"
    assert answer.insufficient_evidence is False
    assert len(answer.claims) >= 1
    assert answer.claims[0].support_status == SupportStatus.SUPPORTED
    assert len(answer.citations) >= 1
    assert answer.citations[0].source_id == "2005.11401"
    assert answer.verification.valid is True
    assert answer.metadata.provider == "mock"
    assert answer.metadata.total_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_research_answer_service_empty_evidence():
    empty_resp = ResearchRetrievalResponse(
        query="Obscure unknown question",
        total_candidates=0,
        total_evidence=0,
        candidates=[],
        reranked=[],
        evidence_pack=EvidencePack(query="Obscure unknown question", selected_passages=[]),
        latency_breakdown=LatencyBreakdown(total_retrieval_latency_ms=5.0),
    )
    mock_retrieval_svc = AsyncMock()
    mock_retrieval_svc.retrieve_evidence.return_value = empty_resp

    service = ResearchAnswerService(
        retrieval_service=mock_retrieval_svc,
        llm_provider=MockLLMProvider(),
    )

    req = ResearchAnswerRequest(query="Obscure unknown question")
    answer = await service.answer_question(req)

    assert answer.insufficient_evidence is True
    assert "Insufficient evidence" in answer.summary
    assert len(answer.claims) == 0
    assert len(answer.citations) == 0

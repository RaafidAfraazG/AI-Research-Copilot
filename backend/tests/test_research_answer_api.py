"""
Unit tests for FastAPI endpoint POST /api/v1/research/answer.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from main import app
from synthesis.models import (
    Citation,
    CitationVerificationResult,
    ClaimItem,
    ResearchAnswer,
    ResearchAnswerSection,
    SourceProvenance,
    SupportStatus,
    SynthesisMetadata,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_api_research_answer_endpoint():
    mock_answer = ResearchAnswer(
        question="What is RAG?",
        summary="RAG combines parametric and non-parametric memory.",
        sections=[
            ResearchAnswerSection(
                title="Overview",
                claims=[
                    ClaimItem(
                        claim_id="claim_001",
                        text="RAG combines dense retrieval and seq2seq generation.",
                        evidence_ids=["evidence_001"],
                        citation_ids=["cite_1"],
                        support_status=SupportStatus.SUPPORTED,
                    )
                ],
            )
        ],
        claims=[
            ClaimItem(
                claim_id="claim_001",
                text="RAG combines dense retrieval and seq2seq generation.",
                evidence_ids=["evidence_001"],
                citation_ids=["cite_1"],
                support_status=SupportStatus.SUPPORTED,
            )
        ],
        citations=[
            Citation(
                citation_id="cite_1",
                number=1,
                title="RAG Paper",
                source_type="arxiv",
                source_id="2005.11401",
                url="https://arxiv.org/abs/2005.11401",
                evidence_id="evidence_001",
                formatted_citation="[1] Patrick Lewis et al., RAG Paper (2020)",
            )
        ],
        sources=[
            SourceProvenance(
                source_id="2005.11401",
                source_type="arxiv",
                title="RAG Paper",
                url="https://arxiv.org/abs/2005.11401",
                evidence_count=1,
            )
        ],
        limitations=["Synthesized from scientific literature corpus."],
        verification=CitationVerificationResult(
            valid=True,
            verified_claim_count=1,
            total_claim_count=1,
            grounding_rate=1.0,
        ),
        metadata=SynthesisMetadata(
            provider="mock",
            model="mock-research-model-v1",
            retrieval_latency_ms=10.0,
            generation_latency_ms=5.0,
            verification_latency_ms=0.5,
            total_latency_ms=15.5,
            claim_count=1,
            supported_claim_count=1,
        ),
        insufficient_evidence=False,
    )

    with patch("synthesis.answer_service.ResearchAnswerService.answer_question", new_callable=AsyncMock) as mock_ans:
        mock_ans.return_value = mock_answer

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v1/research/answer",
                json={
                    "query": "What is RAG?",
                    "candidate_k": 20,
                    "evidence_k": 5,
                },
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["question"] == "What is RAG?"
            assert data["summary"] == "RAG combines parametric and non-parametric memory."
            assert len(data["claims"]) == 1
            assert data["claims"][0]["support_status"] == "SUPPORTED"
            assert data["verification"]["valid"] is True
            assert len(data["citations"]) == 1
            assert data["citations"][0]["source_id"] == "2005.11401"
            assert data["metadata"]["provider"] == "mock"

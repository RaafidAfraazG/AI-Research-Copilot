"""
Tests for answer quality, grounding, and citation metrics (Phase 7).
"""
from __future__ import annotations

import pytest

from evaluation.answer_metrics import AnswerEvaluationScore, AnswerMetricsEvaluator
from evaluation.metrics import (
    compute_distribution_stats,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
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


def test_recall_and_precision_metrics() -> None:
    relevant = {"doc1", "doc2"}
    retrieved = ["doc1", "doc3", "doc4", "doc2", "doc5"]

    assert recall_at_k(retrieved, relevant, k=1) == 0.5
    assert recall_at_k(retrieved, relevant, k=3) == 0.5
    assert recall_at_k(retrieved, relevant, k=4) == 1.0
    assert recall_at_k(retrieved, relevant, k=10) == 1.0

    assert precision_at_k(retrieved, relevant, k=1) == 1.0
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert precision_at_k(retrieved, relevant, k=4) == 0.5
    assert precision_at_k(retrieved, relevant, k=5) == 0.4

    # Edge cases
    assert recall_at_k([], relevant, k=5) == 0.0
    assert recall_at_k(retrieved, set(), k=5) == 0.0
    assert precision_at_k([], relevant, k=5) == 0.0
    assert precision_at_k(retrieved, set(), k=5) == 0.0


def test_mrr_and_ndcg_metrics() -> None:
    relevant = {"doc_target"}
    
    # Target at rank 1
    assert mrr(["doc_target", "doc_other"], relevant) == 1.0
    # Target at rank 2
    assert mrr(["doc_other", "doc_target"], relevant) == 0.5
    # Target at rank 4
    assert mrr(["d1", "d2", "d3", "doc_target"], relevant) == 0.25
    # Target not found
    assert mrr(["d1", "d2"], relevant) == 0.0

    # NDCG
    assert ndcg_at_k(["doc_target", "d1"], relevant, k=5) == 1.0
    assert 0.0 < ndcg_at_k(["d1", "doc_target"], relevant, k=5) < 1.0
    assert ndcg_at_k(["d1", "d2"], relevant, k=5) == 0.0


def test_distribution_statistics() -> None:
    values = [0.2, 0.4, 0.6, 0.8, 1.0]
    stats = compute_distribution_stats(values)

    assert stats.mean == 0.6
    assert stats.median == 0.6
    assert stats.min_val == 0.2
    assert stats.max_val == 1.0
    assert stats.count == 5
    assert stats.std_dev > 0.0

    # Empty list edge case
    empty_stats = compute_distribution_stats([])
    assert empty_stats.mean == 0.0
    assert empty_stats.count == 0


def test_answer_metrics_evaluator_fully_supported() -> None:
    answer = ResearchAnswer(
        question="What is RAG?",
        summary="RAG combines parametric and non-parametric memory [1]. It uses dense retrieval [2].",
        sections=[
            ResearchAnswerSection(
                title="Overview",
                claims=[
                    ClaimItem(
                        claim_id="c1",
                        text="RAG combines parametric and non-parametric memory.",
                        evidence_ids=["evidence_001"],
                        citation_ids=["cite_1"],
                        support_status=SupportStatus.SUPPORTED,
                    ),
                    ClaimItem(
                        claim_id="c2",
                        text="It uses dense retrieval.",
                        evidence_ids=["evidence_002"],
                        citation_ids=["cite_2"],
                        support_status=SupportStatus.SUPPORTED,
                    ),
                ],
            )
        ],
        claims=[
            ClaimItem(
                claim_id="c1",
                text="RAG combines parametric and non-parametric memory.",
                evidence_ids=["evidence_001"],
                citation_ids=["cite_1"],
                support_status=SupportStatus.SUPPORTED,
            ),
            ClaimItem(
                claim_id="c2",
                text="It uses dense retrieval.",
                evidence_ids=["evidence_002"],
                citation_ids=["cite_2"],
                support_status=SupportStatus.SUPPORTED,
            ),
        ],
        citations=[
            Citation(
                citation_id="cite_1",
                number=1,
                title="RAG Paper",
                source_type="arxiv",
                source_id="arxiv:2005.11401",
                evidence_id="evidence_001",
                formatted_citation="[1] Lewis et al. RAG Paper.",
                url="https://arxiv.org/abs/2005.11401",
            ),
            Citation(
                citation_id="cite_2",
                number=2,
                title="DPR Paper",
                source_type="arxiv",
                source_id="arxiv:2004.04906",
                evidence_id="evidence_002",
                formatted_citation="[2] Karpukhin et al. DPR Paper.",
                url="https://arxiv.org/abs/2004.04906",
            ),
        ],
        sources=[
            SourceProvenance(
                source_id="arxiv:2005.11401",
                source_type="arxiv",
                title="RAG Paper",
                url="https://arxiv.org/abs/2005.11401",
                evidence_count=1,
            ),
            SourceProvenance(
                source_id="arxiv:2004.04906",
                source_type="arxiv",
                title="DPR Paper",
                url="https://arxiv.org/abs/2004.04906",
                evidence_count=1,
            ),
        ],
        verification=CitationVerificationResult(
            valid=True,
            verified_claim_count=2,
            total_claim_count=2,
            grounding_rate=1.0,
            unsupported_claim_ids=[],
            errors=[],
            warnings=[],
        ),
        metadata=SynthesisMetadata(
            provider="mock",
            model="mock-deterministic",
            total_latency_ms=10.0,
        ),
    )

    score = AnswerMetricsEvaluator.evaluate_answer(
        answer=answer,
        query_id="q001",
        expected_dimensions=["parametric memory", "dense retrieval"],
        relevant_documents=["2005.11401", "2004.04906"],
    )

    assert score.query_id == "q001"
    assert score.has_claims is True
    assert score.total_claims == 2
    assert score.supported_claims == 2
    assert score.unsupported_claims == 0
    assert score.claim_support_rate == 1.0
    assert score.unsupported_claim_rate == 0.0
    assert score.citation_coverage == 1.0
    assert score.citation_validity_rate == 1.0
    assert score.citation_precision == 1.0
    assert score.citation_traceability_rate == 1.0
    assert score.unique_cited_documents == 2
    assert score.unique_cited_sources == 2
    assert score.evidence_dimension_coverage == 1.0


def test_answer_metrics_evaluator_zero_claims_safe() -> None:
    answer = ResearchAnswer(
        question="What is an unknown topic?",
        summary="I have insufficient evidence to answer this question.",
        insufficient_evidence=True,
        claims=[],
        citations=[],
        sources=[],
        verification=CitationVerificationResult(
            valid=True,
            verified_claim_count=0,
            total_claim_count=0,
            grounding_rate=0.0,
        ),
        metadata=SynthesisMetadata(
            provider="mock",
            model="mock",
        ),
    )

    score = AnswerMetricsEvaluator.evaluate_answer(
        answer=answer,
        query_id="q_empty",
        expected_dimensions=["unknown dimension"],
    )

    assert score.has_claims is False
    assert score.total_claims == 0
    assert score.claim_support_rate == 0.0
    assert score.unsupported_claim_rate == 0.0
    assert score.citation_coverage == 0.0
    assert score.citation_validity_rate == 0.0
    assert score.insufficient_evidence is True


def test_answer_metrics_evaluator_with_unsupported_claim() -> None:
    answer = ResearchAnswer(
        question="Explain transformer mechanisms",
        summary="Transformers use self-attention [1]. Transformers also predict the stock market with 100% accuracy.",
        claims=[
            ClaimItem(
                claim_id="c1",
                text="Transformers use self-attention.",
                evidence_ids=["evidence_001"],
                citation_ids=["cite_1"],
                support_status=SupportStatus.SUPPORTED,
            ),
            ClaimItem(
                claim_id="c2",
                text="Transformers predict the stock market with 100% accuracy.",
                evidence_ids=[],
                citation_ids=[],
                support_status=SupportStatus.UNSUPPORTED,
            ),
        ],
        citations=[
            Citation(
                citation_id="cite_1",
                number=1,
                title="Attention Paper",
                source_type="arxiv",
                source_id="arxiv:1706.03762",
                evidence_id="evidence_001",
                formatted_citation="[1] Vaswani et al.",
                url="https://arxiv.org/abs/1706.03762",
            )
        ],
        sources=[
            SourceProvenance(
                source_id="arxiv:1706.03762",
                source_type="arxiv",
                title="Attention Paper",
                url="https://arxiv.org/abs/1706.03762",
                evidence_count=1,
            )
        ],
        verification=CitationVerificationResult(
            valid=False,
            verified_claim_count=1,
            total_claim_count=2,
            grounding_rate=0.5,
            unsupported_claim_ids=["c2"],
        ),
        metadata=SynthesisMetadata(
            provider="mock",
            model="mock",
        ),
    )

    score = AnswerMetricsEvaluator.evaluate_answer(
        answer=answer,
        query_id="q_partial",
        expected_dimensions=["self-attention", "stock prediction"],
    )

    assert score.total_claims == 2
    assert score.supported_claims == 1
    assert score.unsupported_claims == 1
    assert score.claim_support_rate == 0.5
    assert score.unsupported_claim_rate == 0.5
    assert score.citation_coverage == 0.5
    assert score.citation_validity_rate == 1.0

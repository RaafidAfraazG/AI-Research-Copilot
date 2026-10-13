"""
Unit tests for CitationVerifier and grounding policy audits.
"""
from __future__ import annotations

import pytest

from ranking.evidence import EvidencePack, EvidencePassage
from synthesis.citation_verifier import CitationVerifier
from synthesis.models import Citation, ClaimItem, SupportStatus


@pytest.fixture
def sample_evidence_setup():
    passages = {
        "evidence_001": EvidencePassage(
            passage_id="p-01",
            document_id="doc-01",
            text="RAG combines parametric memory.",
            title="RAG Paper",
            source_type="arxiv",
            source_id="2005.11401",
            relevance_score=9.0,
            rank=1,
        ),
        "evidence_002": EvidencePassage(
            passage_id="p-02",
            document_id="doc-02",
            text="DPR uses dual encoders.",
            title="DPR Paper",
            source_type="arxiv",
            source_id="2004.04906",
            relevance_score=8.5,
            rank=2,
        ),
    }

    pack = EvidencePack(
        query="What is RAG?",
        selected_passages=list(passages.values()),
        total_candidates=5,
        total_evidence=2,
    )

    citations = [
        Citation(
            citation_id="cite_1",
            number=1,
            title="RAG Paper",
            source_type="arxiv",
            source_id="2005.11401",
            evidence_id="evidence_001",
            formatted_citation="[1] RAG Paper",
        ),
        Citation(
            citation_id="cite_2",
            number=2,
            title="DPR Paper",
            source_type="arxiv",
            source_id="2004.04906",
            evidence_id="evidence_002",
            formatted_citation="[2] DPR Paper",
        ),
    ]

    return passages, pack, citations


def test_verifier_all_claims_supported(sample_evidence_setup):
    passages, pack, citations = sample_evidence_setup
    claims = [
        ClaimItem(
            claim_id="claim_001",
            text="RAG combines parametric and non-parametric memory.",
            evidence_ids=["evidence_001"],
            citation_ids=["cite_1"],
        ),
        ClaimItem(
            claim_id="claim_002",
            text="DPR performs dual encoder search.",
            evidence_ids=["evidence_002"],
            citation_ids=["cite_2"],
        ),
    ]

    res, lat = CitationVerifier.verify(claims, citations, passages, pack)

    assert res.valid is True
    assert res.verified_claim_count == 2
    assert res.total_claim_count == 2
    assert res.grounding_rate == 1.0
    assert len(res.errors) == 0
    assert claims[0].support_status == SupportStatus.SUPPORTED
    assert claims[1].support_status == SupportStatus.SUPPORTED
    assert lat >= 0.0


def test_verifier_detects_hallucinated_evidence_id(sample_evidence_setup):
    passages, pack, citations = sample_evidence_setup
    claims = [
        ClaimItem(
            claim_id="claim_001",
            text="Hallucinated fact.",
            evidence_ids=["evidence_999"],  # Does not exist!
            citation_ids=["cite_1"],
        ),
    ]

    res, _ = CitationVerifier.verify(claims, citations, passages, pack)

    assert res.valid is False
    assert len(res.errors) >= 1
    assert "evidence_999" in res.errors[0]
    assert claims[0].support_status == SupportStatus.UNSUPPORTED
    assert "claim_001" in res.unsupported_claim_ids


def test_verifier_detects_partially_supported_claim(sample_evidence_setup):
    passages, pack, citations = sample_evidence_setup
    claims = [
        ClaimItem(
            claim_id="claim_001",
            text="One real fact and one hallucinated fact.",
            evidence_ids=["evidence_001", "evidence_999"],  # 1 valid, 1 invalid
            citation_ids=["cite_1"],
        ),
    ]

    res, _ = CitationVerifier.verify(claims, citations, passages, pack)

    assert claims[0].support_status == SupportStatus.PARTIALLY_SUPPORTED
    assert res.valid is False


def test_verifier_detects_missing_evidence_on_claim(sample_evidence_setup):
    passages, pack, citations = sample_evidence_setup
    claims = [
        ClaimItem(
            claim_id="claim_001",
            text="Claim with empty evidence list.",
            evidence_ids=[],
        ),
    ]

    res, _ = CitationVerifier.verify(claims, citations, passages, pack)

    assert claims[0].support_status == SupportStatus.UNSUPPORTED
    assert "claim_001" in res.unsupported_claim_ids
    assert res.valid is False


def test_verifier_orphan_citation_warning(sample_evidence_setup):
    passages, pack, citations = sample_evidence_setup
    # Only claim_001 uses cite_1, cite_2 is left as an orphan
    claims = [
        ClaimItem(
            claim_id="claim_001",
            text="Fact using only cite 1.",
            evidence_ids=["evidence_001"],
            citation_ids=["cite_1"],
        ),
    ]

    res, _ = CitationVerifier.verify(claims, citations, passages, pack)

    assert res.valid is True  # Orphan citations generate warnings, not invalidating grounding
    assert len(res.warnings) >= 1
    assert "cite_2" in res.warnings[0]


def test_verifier_empty_evidence_pack():
    empty_pack = EvidencePack(query="What is AI?", selected_passages=[])
    res, _ = CitationVerifier.verify(claims=[], citations=[], id_to_passage={}, evidence_pack=empty_pack)

    assert res.valid is False
    assert any("zero passages" in e for e in res.errors)

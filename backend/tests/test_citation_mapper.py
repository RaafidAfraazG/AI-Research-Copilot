"""
Unit tests for CitationMapper (citation formatting and claim mapping).
"""
from __future__ import annotations

from ranking.evidence import EvidencePassage
from synthesis.citation_mapper import CitationMapper
from synthesis.models import ClaimItem


def test_format_citation_string():
    # Single author
    c1 = CitationMapper.format_citation_string(
        number=1,
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        source_type="arxiv",
        source_id="1706.03762",
        publication_date="2017-06-12",
        url="https://arxiv.org/abs/1706.03762",
    )
    assert "[1] Ashish Vaswani, \"Attention Is All You Need\" (2017), arXiv:1706.03762." in c1
    assert "<https://arxiv.org/abs/1706.03762>" in c1

    # Multiple authors
    c2 = CitationMapper.format_citation_string(
        number=2,
        title="RAG Paper",
        authors=["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus"],
        source_type="arxiv",
        source_id="2005.11401",
    )
    assert "[2] Patrick Lewis et al., \"RAG Paper\", arXiv:2005.11401." in c2


def test_map_citations_to_claims():
    passages = {
        "evidence_001": EvidencePassage(
            passage_id="p-01",
            document_id="doc-01",
            text="Text 1",
            title="Paper One",
            source_type="arxiv",
            source_id="2005.11401",
            url="https://arxiv.org/abs/2005.11401",
            relevance_score=9.0,
            rank=1,
            metadata={"authors": ["Alice Smith", "Bob Jones"], "year": 2020},
        ),
        "evidence_002": EvidencePassage(
            passage_id="p-02",
            document_id="doc-02",
            text="Text 2",
            title="Paper Two",
            source_type="web",
            source_id="https://example.com/paper",
            url="https://example.com/paper",
            relevance_score=8.5,
            rank=2,
        ),
    }

    claims = [
        ClaimItem(claim_id="claim_001", text="Finding 1", evidence_ids=["evidence_001"]),
        ClaimItem(claim_id="claim_002", text="Finding 2", evidence_ids=["evidence_001", "evidence_002"]),
    ]

    citations, sources = CitationMapper.map_citations(claims, passages)

    assert len(citations) == 2
    assert citations[0].citation_id == "cite_1"
    assert citations[1].citation_id == "cite_2"

    assert claims[0].citation_ids == ["cite_1"]
    assert claims[1].citation_ids == ["cite_1", "cite_2"]

    assert len(sources) == 2
    assert any(s.source_id == "2005.11401" for s in sources)

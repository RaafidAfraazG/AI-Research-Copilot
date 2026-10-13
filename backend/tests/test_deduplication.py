"""
Unit tests for DocumentDeduplicator and canonical URL normalization.
"""
from __future__ import annotations

import pytest

from connectors.deduplication import (
    DocumentDeduplicator,
    extract_canonical_id,
    normalize_url,
)
from connectors.models import CitationInfo, ProvenanceInfo, ResearchDocument


def test_normalize_url():
    """Verify URL normalization removes tracking params, normalizes slashes, and lowercases host."""
    url1 = "https://EXAMPLE.com/path/to/paper/?utm_source=twitter&utm_medium=social&b=2&a=1#section"
    norm1 = normalize_url(url1)
    assert norm1 == "https://example.com/path/to/paper?a=1&b=2"

    url2 = "http://arxiv.org:80/abs/2005.11401v2///"
    norm2 = normalize_url(url2)
    assert norm2 == "http://arxiv.org/abs/2005.11401v2"


def test_extract_canonical_id_priorities():
    """Verify canonical ID resolution prioritizes arXiv ID, then DOI, then URL."""
    # 1. arXiv paper
    arxiv_doc = ResearchDocument(
        source_type="arxiv",
        source_id="2005.11401v2",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        citation_info=CitationInfo(arxiv_id="2005.11401v2"),
        url="https://arxiv.org/abs/2005.11401v2",
    )
    assert extract_canonical_id(arxiv_doc) == "arxiv:2005.11401"

    # 2. Semantic Scholar paper with arXiv ID in cross_source_ids
    s2_doc = ResearchDocument(
        source_type="semantic_scholar",
        source_id="s2_abc123",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        citation_info=CitationInfo(arxiv_id="2005.11401"),
        provenance=ProvenanceInfo(
            source_type="semantic_scholar",
            source_id="s2_abc123",
            cross_source_ids={"arxiv": "2005.11401", "doi": "10.1234/5678"},
        ),
    )
    assert extract_canonical_id(s2_doc) == "arxiv:2005.11401"

    # 3. Web document with DOI
    doi_doc = ResearchDocument(
        source_type="web",
        source_id="https://example.org/paper",
        title="Some Paper",
        citation_info=CitationInfo(doi="10.1145/3340531.3412007"),
    )
    assert extract_canonical_id(doi_doc) == "doi:10.1145/3340531.3412007"

    # 4. Standard Web page
    web_doc = ResearchDocument(
        source_type="web",
        source_id="https://example.com/doc",
        title="Doc",
        url="https://example.com/doc?utm_source=feed",
    )
    assert extract_canonical_id(web_doc) == "url:https://example.com/doc"


def test_deduplicator_cross_source_merging():
    """Verify deduplicator detects matching arXiv & S2 documents and merges metadata/provenance."""
    dedup = DocumentDeduplicator()

    # Ingest arXiv first
    arxiv_doc = ResearchDocument(
        source_type="arxiv",
        source_id="2005.11401",
        title="Retrieval-Augmented Generation",
        authors=["Patrick Lewis", "Ethan Perez"],
        abstract="We present RAG models...",
        url="https://arxiv.org/abs/2005.11401",
        citation_info=CitationInfo(arxiv_id="2005.11401"),
        provenance=ProvenanceInfo(
            source_type="arxiv",
            source_id="2005.11401",
            canonical_url="https://arxiv.org/abs/2005.11401",
        ),
    )
    res1, is_new1 = dedup.register_and_merge(arxiv_doc)
    assert is_new1 is True
    assert res1.citation_info.citation_count is None

    # Ingest Semantic Scholar second with richer citation metrics
    s2_doc = ResearchDocument(
        source_type="semantic_scholar",
        source_id="s2_rag_id",
        title="Retrieval-Augmented Generation for NLP Tasks",
        authors=["Patrick Lewis", "Aleksandra Piktus", "Ethan Perez"],
        abstract="We present RAG models...",
        url="https://www.semanticscholar.org/paper/s2_rag_id",
        citation_info=CitationInfo(
            arxiv_id="2005.11401",
            citation_count=4500,
            tldr="This paper introduces RAG combining parametric and non-parametric memory.",
        ),
        provenance=ProvenanceInfo(
            source_type="semantic_scholar",
            source_id="s2_rag_id",
            canonical_url="https://www.semanticscholar.org/paper/s2_rag_id",
            cross_source_ids={"arxiv": "2005.11401"},
        ),
    )

    res2, is_new2 = dedup.register_and_merge(s2_doc)
    assert is_new2 is False  # Merged into existing
    assert res2.citation_info.citation_count == 4500
    assert "introduces RAG" in (res2.citation_info.tldr or "")
    assert len(res2.authors) == 3
    assert res2.provenance.cross_source_ids["arxiv"] == "2005.11401"
    assert res2.provenance.cross_source_ids["semantic_scholar"] == "s2_rag_id"


def test_deduplicator_content_hash_collision():
    """Verify duplicate documents with exact same title and abstract are detected."""
    dedup = DocumentDeduplicator()

    doc_a = ResearchDocument(
        source_type="web",
        source_id="https://site-a.com/page1",
        title="Identical Study On AI",
        abstract="This is the exact same text abstract.",
    )
    doc_b = ResearchDocument(
        source_type="web",
        source_id="https://site-b.com/page2",
        title="Identical Study On AI",
        abstract="This is the exact same text abstract.",
    )

    _, is_new_a = dedup.register_and_merge(doc_a)
    assert is_new_a is True

    _, is_new_b = dedup.register_and_merge(doc_b)
    assert is_new_b is False  # Detected identical content hash

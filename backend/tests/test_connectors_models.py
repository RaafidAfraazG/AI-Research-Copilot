"""
Unit tests for Connector Models and Normalization (connectors/models.py).
"""
from __future__ import annotations

import pytest

from connectors.models import (
    CitationInfo,
    ConnectorSearchRequest,
    ConnectorSearchResponse,
    ProvenanceInfo,
    ResearchDocument,
)


class TestConnectorModels:
    def test_research_document_creation(self) -> None:
        doc = ResearchDocument(
            source_type="arxiv",
            source_id="2005.11401",
            title="Retrieval-Augmented Generation",
            authors=["Patrick Lewis", "Ethan Perez"],
            abstract="RAG combines parametric and non-parametric memory.",
            url="https://arxiv.org/abs/2005.11401",
            pdf_url="https://arxiv.org/pdf/2005.11401.pdf",
            publication_date="2020-05-22",
            categories=["cs.CL", "cs.AI"],
            citation_info=CitationInfo(citation_count=1500, arxiv_id="2005.11401"),
        )
        assert doc.source_type == "arxiv"
        assert doc.source_id == "2005.11401"
        assert len(doc.authors) == 2
        assert doc.citation_info.citation_count == 1500

    def test_content_hash_deterministic(self) -> None:
        doc1 = ResearchDocument(
            source_type="arxiv",
            source_id="2005.11401",
            title="Retrieval-Augmented Generation",
            abstract="RAG combines parametric and non-parametric memory.",
        )
        doc2 = ResearchDocument(
            source_type="semantic_scholar",
            source_id="s2-12345",
            title="Retrieval-Augmented Generation",
            abstract="RAG combines parametric and non-parametric memory.",
        )
        # Content hashes should match because indexable content (abstract) is identical
        assert doc1.compute_content_hash() == doc2.compute_content_hash()
        assert len(doc1.compute_content_hash()) == 64

    def test_to_ingest_request_preserves_metadata(self) -> None:
        doc = ResearchDocument(
            source_type="arxiv",
            source_id="2005.11401",
            title="Retrieval-Augmented Generation",
            authors=["Patrick Lewis"],
            abstract="RAG abstract text.",
            full_text="Full paper text goes here.",
            url="https://arxiv.org/abs/2005.11401",
            venue="NeurIPS 2020",
            categories=["cs.CL"],
            citation_info=CitationInfo(citation_count=1500, tldr="RAG models combine retrieval with seq2seq"),
        )
        req = doc.to_ingest_request(chunk_size_tokens=256, chunk_overlap_tokens=30)
        assert req.source_type == "arxiv"
        assert req.source_id == "2005.11401"
        assert req.title == "Retrieval-Augmented Generation"
        assert "RAG abstract text" in req.content
        assert "Full paper text" in req.content
        assert req.metadata["venue"] == "NeurIPS 2020"
        assert req.metadata["citation_count"] == 1500
        assert req.metadata["tldr"] == "RAG models combine retrieval with seq2seq"
        assert req.chunk_size_tokens == 256
        assert req.chunk_overlap_tokens == 30

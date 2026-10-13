"""
Unit tests for EvidenceSelector (diversity, deduplication, thresholding).
"""
from __future__ import annotations

import pytest

from ranking.evidence_selector import EvidenceSelector, compute_token_jaccard
from retrieval.models import ScoredChunk


def test_compute_token_jaccard():
    t1 = "Retrieval augmented generation for question answering"
    t2 = "Retrieval augmented generation for question answering systems"
    t3 = "Completely distinct unrelated text content"

    sim_high = compute_token_jaccard(t1, t2)
    sim_low = compute_token_jaccard(t1, t3)

    assert sim_high > 0.8
    assert sim_low == 0.0


def test_evidence_selector_max_evidence():
    chunks = [
        ScoredChunk(
            chunk_id=f"c-{i}",
            document_id=f"d-{i}",
            source_type="arxiv",
            source_id=f"id-{i}",
            content=f"Unique text content for passage number {i}",
            rerank_score=10.0 - i,
            score=10.0 - i,
            rank=i + 1,
        )
        for i in range(10)
    ]

    selector = EvidenceSelector(max_evidence=3)
    pack, lat = selector.select("Query", chunks)

    assert len(pack.selected_passages) == 3
    assert pack.total_evidence == 3
    assert pack.total_candidates == 10
    assert pack.selected_passages[0].passage_id == "c-0"
    assert pack.selected_passages[2].passage_id == "c-2"
    assert lat >= 0.0


def test_evidence_selector_min_score_threshold():
    chunks = [
        ScoredChunk(
            chunk_id="c-1",
            document_id="d-1",
            source_type="arxiv",
            source_id="1",
            content="Passage 1 above threshold",
            rerank_score=5.5,
            score=5.5,
            rank=1,
        ),
        ScoredChunk(
            chunk_id="c-2",
            document_id="d-2",
            source_type="arxiv",
            source_id="2",
            content="Passage 2 below threshold",
            rerank_score=1.2,
            score=1.2,
            rank=2,
        ),
    ]

    selector = EvidenceSelector(min_score=3.0)
    pack, _ = selector.select("Query", chunks)

    assert len(pack.selected_passages) == 1
    assert pack.selected_passages[0].passage_id == "c-1"


def test_evidence_selector_document_diversity_quota():
    """Verify max_per_document limit prevents one document from dominating the evidence pack."""
    chunks = [
        ScoredChunk(
            chunk_id=f"doc1-chunk-{i}",
            document_id="doc-1",
            source_type="arxiv",
            source_id="2005.11401",
            content=f"Distinct section {i} from the first document on RAG architecture.",
            rerank_score=9.0 - (i * 0.1),
            score=9.0 - (i * 0.1),
            rank=i + 1,
        )
        for i in range(5)
    ]
    # Add a chunk from doc-2
    chunks.append(
        ScoredChunk(
            chunk_id="doc2-chunk-0",
            document_id="doc-2",
            source_type="arxiv",
            source_id="2004.04906",
            content="Distinct content from the second document on DPR dense retrieval.",
            rerank_score=8.0,
            score=8.0,
            rank=6,
        )
    )

    selector = EvidenceSelector(max_evidence=4, max_per_document=2)
    pack, _ = selector.select("Query", chunks)

    # Should select 2 from doc-1 and 1 from doc-2 (total 3)
    doc1_selected = [p for p in pack.selected_passages if p.document_id == "doc-1"]
    doc2_selected = [p for p in pack.selected_passages if p.document_id == "doc-2"]

    assert len(doc1_selected) == 2
    assert len(doc2_selected) == 1
    assert len(pack.selected_passages) == 3


def test_evidence_selector_near_duplicate_filtering():
    """Verify near-duplicate passages are filtered out."""
    chunks = [
        ScoredChunk(
            chunk_id="c-1",
            document_id="d-1",
            source_type="web",
            source_id="https://example.com/1",
            content="Retrieval-Augmented Generation models combine parametric and non-parametric memory effectively.",
            rerank_score=8.5,
            score=8.5,
            rank=1,
        ),
        ScoredChunk(
            chunk_id="c-2",
            document_id="d-2",
            source_type="web",
            source_id="https://example.com/2",
            # Near identical text from different doc/source
            content="Retrieval-Augmented Generation models combine parametric and non-parametric memory effectively!",
            rerank_score=8.4,
            score=8.4,
            rank=2,
        ),
        ScoredChunk(
            chunk_id="c-3",
            document_id="d-3",
            source_type="web",
            source_id="https://example.com/3",
            content="Dense vector retrieval uses approximate nearest neighbors for fast semantic indexing.",
            rerank_score=7.0,
            score=7.0,
            rank=3,
        ),
    ]

    selector = EvidenceSelector(max_evidence=3, similarity_threshold=0.85)
    pack, _ = selector.select("Query", chunks)

    assert len(pack.selected_passages) == 2
    assert pack.selected_passages[0].passage_id == "c-1"
    assert pack.selected_passages[1].passage_id == "c-3"

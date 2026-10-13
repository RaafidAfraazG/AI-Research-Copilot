"""
Unit tests for EvidencePassage and EvidencePack schemas and serialization.
"""
from __future__ import annotations

import json
from ranking.evidence import EvidencePack, EvidencePassage


def test_evidence_passage_model():
    passage = EvidencePassage(
        passage_id="chk-123",
        document_id="doc-456",
        text="Sample evidence text.",
        title="RAG Systems",
        source_type="arxiv",
        source_id="2005.11401",
        url="https://arxiv.org/abs/2005.11401",
        relevance_score=8.75,
        rank=1,
        provenance={"authors": ["Patrick Lewis"]},
    )

    assert passage.passage_id == "chk-123"
    assert passage.relevance_score == 8.75
    assert passage.url == "https://arxiv.org/abs/2005.11401"


def test_evidence_pack_json_serialization():
    pack = EvidencePack(
        query="What is RAG?",
        selected_passages=[
            EvidencePassage(
                passage_id="p-1",
                document_id="d-1",
                text="RAG combines parametric memory.",
                title="RAG Paper",
                source_type="arxiv",
                source_id="2005.11401",
                relevance_score=9.1,
                rank=1,
            )
        ],
        total_candidates=15,
        total_evidence=1,
        retrieval_metadata={"fusion": "rrf"},
        ranking_metadata={"model": "ms-marco-MiniLM"},
    )

    data = pack.model_dump()
    assert data["query"] == "What is RAG?"
    assert data["total_candidates"] == 15
    assert len(data["selected_passages"]) == 1

    # Verify JSON serialization works without error
    json_str = pack.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["selected_passages"][0]["relevance_score"] == 9.1

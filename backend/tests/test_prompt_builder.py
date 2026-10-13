"""
Unit tests for GroundedPromptBuilder.
"""
from __future__ import annotations

from ranking.evidence import EvidencePack, EvidencePassage
from synthesis.prompt_builder import GroundedPromptBuilder


def test_prompt_builder_formats_evidence_blocks():
    pack = EvidencePack(
        query="What is RAG?",
        selected_passages=[
            EvidencePassage(
                passage_id="chk-01",
                document_id="doc-01",
                text="RAG combines parametric memory with non-parametric dense index.",
                title="RAG Paper",
                source_type="arxiv",
                source_id="2005.11401",
                url="https://arxiv.org/abs/2005.11401",
                relevance_score=8.95,
                rank=1,
            ),
            EvidencePassage(
                passage_id="chk-02",
                document_id="doc-02",
                text="DPR indexes Wikipedia passages using dual BERT encoders.",
                title="DPR Paper",
                source_type="arxiv",
                source_id="2004.04906",
                url="https://arxiv.org/abs/2004.04906",
                relevance_score=7.40,
                rank=2,
            ),
        ],
        total_candidates=10,
        total_evidence=2,
    )

    ev_text, id_map = GroundedPromptBuilder.format_evidence_block(pack)

    assert "evidence_001" in id_map
    assert "evidence_002" in id_map
    assert id_map["evidence_001"].passage_id == "chk-01"
    assert id_map["evidence_002"].passage_id == "chk-02"

    assert "[EVIDENCE_ID: evidence_001]" in ev_text
    assert "[EVIDENCE_ID: evidence_002]" in ev_text
    assert "https://arxiv.org/abs/2005.11401" in ev_text

    user_prompt = GroundedPromptBuilder.build_user_prompt("What is RAG?", pack)
    assert "USER RESEARCH QUESTION:\nWhat is RAG?" in user_prompt
    assert "RETRIEVED EVIDENCE PASSAGES" in user_prompt

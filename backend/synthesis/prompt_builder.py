"""
Prompt Builder for strictly grounded research synthesis.
Formats EvidencePack into structured delimited blocks with explicit citation constraints.
"""
from __future__ import annotations

from typing import Any
from ranking.evidence import EvidencePack, EvidencePassage


class GroundedPromptBuilder:
    """
    Constructs system and user prompts enforcing evidence-only grounded synthesis.
    """

    SYSTEM_PROMPT = """You are an expert Scientific Research Synthesis Assistant.
Your task is to synthesize a precise, authoritative, and strictly grounded research answer based ONLY on the provided retrieved evidence passages.

CRITICAL GROUNDING RULES:
1. Grounding: Answer ONLY using facts explicitly stated in the RETRIEVED EVIDENCE. Do NOT use outside knowledge.
2. Fact Attribution: Every factual statement or claim MUST reference one or more supporting EVIDENCE_IDs (e.g. ["evidence_001"]).
3. Truthfulness: Never invent facts, author names, benchmarks, dates, or citations.
4. Insufficient Evidence: If the evidence is empty or does not contain enough information to answer the question, state clearly in summary: "Insufficient evidence to answer this reliably from the retrieved sources." and set "insufficient_evidence": true.
5. Limitations: Explicitly list any gaps, limitations, or conflicts noted in the evidence.
6. Format: You MUST output valid JSON conforming strictly to the requested schema. Do NOT enclose in backticks or markdown code fences if JSON mode is active.

REQUIRED JSON OUTPUT SCHEMA:
{
  "summary": "High-level 2-3 sentence executive summary answering the question.",
  "insufficient_evidence": false,
  "sections": [
    {
      "title": "Section Title",
      "claims": [
        {
          "claim_id": "claim_001",
          "text": "Factual assertion text.",
          "evidence_ids": ["evidence_001"]
        }
      ]
    }
  ],
  "claims": [
    {
      "claim_id": "claim_001",
      "text": "Factual assertion text.",
      "evidence_ids": ["evidence_001"]
    }
  ],
  "limitations": [
    "Known limitation or boundary condition identified in evidence."
  ]
}
"""

    @classmethod
    def format_evidence_block(cls, evidence_pack: EvidencePack) -> tuple[str, dict[str, EvidencePassage]]:
        """
        Format EvidencePassages into numbered evidence blocks (evidence_001, evidence_002, ...)
        and return mapping of evidence_id -> EvidencePassage.
        """
        id_to_passage: dict[str, EvidencePassage] = {}
        formatted_blocks: list[str] = []

        for idx, passage in enumerate(evidence_pack.selected_passages, start=1):
            evidence_id = f"evidence_{idx:03d}"
            id_to_passage[evidence_id] = passage

            block = (
                f"[EVIDENCE_ID: {evidence_id}]\n"
                f"Document ID: {passage.document_id}\n"
                f"Source: {passage.source_type}:{passage.source_id}\n"
                f"Title: {passage.title}\n"
                f"URL: {passage.url or 'N/A'}\n"
                f"Relevance Score: {passage.relevance_score}\n"
                f"Passage Content:\n{passage.text}\n"
            )
            formatted_blocks.append(block)

        full_evidence_text = "\n".join(formatted_blocks) if formatted_blocks else "NO EVIDENCE PASSAGES AVAILABLE."
        return full_evidence_text, id_to_passage

    @classmethod
    def build_user_prompt(cls, question: str, evidence_pack: EvidencePack) -> str:
        """
        Assemble the complete user prompt containing the research question and structured evidence.
        """
        evidence_text, _ = cls.format_evidence_block(evidence_pack)

        user_prompt = f"""USER RESEARCH QUESTION:
{question}

RETRIEVED EVIDENCE PASSAGES ({len(evidence_pack.selected_passages)} passages available):
--------------------------------------------------------------------------------
{evidence_text}
--------------------------------------------------------------------------------

Synthesize a comprehensive, grounded research answer in valid JSON strictly according to the system instructions.
"""
        return user_prompt

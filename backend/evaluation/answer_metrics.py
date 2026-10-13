"""
Answer Quality, Groundedness, and Citation Evaluation Metrics.
Provides deterministic, reproducible measurement of factual claim support,
citation precision, provenance traceability, and evidence dimension coverage.
"""
from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field

from synthesis.models import ResearchAnswer, SupportStatus


class AnswerEvaluationScore(BaseModel):
    """
    Structured metrics evaluating the factual grounding, citation validity,
    and dimension coverage of a generated research answer.
    """
    query_id: str = Field(default="", description="Evaluation query identifier")
    question: str = Field(default="", description="Original user research question")
    has_claims: bool = Field(default=False, description="True if at least one factual claim was generated")
    total_claims: int = Field(default=0, description="Total factual assertions evaluated")
    supported_claims: int = Field(default=0, description="Number of fully supported claims")
    partially_supported_claims: int = Field(default=0, description="Number of partially supported claims")
    unsupported_claims: int = Field(default=0, description="Number of unsupported claims")
    claim_support_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of supported claims to total claims")
    unsupported_claim_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of unsupported claims to total claims")
    total_citations: int = Field(default=0, description="Total rendered bibliographic citations")
    citation_coverage: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of claims having at least one citation")
    citation_validity_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of citations referencing valid evidence IDs")
    citation_precision: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of citations matching authentic source metadata")
    citation_traceability_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of claims traceable end-to-end to source URLs")
    unique_cited_documents: int = Field(default=0, description="Count of distinct parent documents cited")
    unique_cited_sources: int = Field(default=0, description="Count of distinct canonical source IDs/URLs cited")
    evidence_dimension_coverage: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of expected research dimensions found in evidence")
    insufficient_evidence: bool = Field(default=False, description="Whether the answer flagged insufficient evidence")
    details: dict[str, Any] = Field(default_factory=dict, description="Audit traces and breakdown")


class AnswerMetricsEvaluator:
    """
    Evaluator assessing answer grounding, claim support, and citation integrity.
    """

    @classmethod
    def evaluate_answer(
        cls,
        answer: ResearchAnswer,
        query_id: str = "",
        expected_dimensions: list[str] | None = None,
        relevant_documents: list[str] | None = None,
    ) -> AnswerEvaluationScore:
        """
        Evaluate a ResearchAnswer object against Groundedness and Citation criteria.
        """
        claims = answer.claims
        citations = answer.citations
        sources = answer.sources
        verif = answer.verification

        total_claims = len(claims)
        has_claims = total_claims > 0

        # Claim support counting
        supported_count = sum(1 for c in claims if c.support_status == SupportStatus.SUPPORTED)
        part_count = sum(1 for c in claims if c.support_status == SupportStatus.PARTIALLY_SUPPORTED)
        unsupp_count = sum(1 for c in claims if c.support_status == SupportStatus.UNSUPPORTED)

        claim_support_rate = round(supported_count / total_claims, 4) if total_claims > 0 else 0.0
        unsupported_claim_rate = round(unsupp_count / total_claims, 4) if total_claims > 0 else 0.0

        # Citation Coverage (fraction of claims with >= 1 citation)
        claims_with_cites = sum(1 for c in claims if len(c.citation_ids) > 0 or len(c.evidence_ids) > 0)
        citation_coverage = round(claims_with_cites / total_claims, 4) if total_claims > 0 else 0.0

        # Citation Validity Rate (from verif audit errors)
        total_citations = len(citations)
        invalid_cite_errors = [e for e in verif.errors if "evidence" in e.lower() or "citation" in e.lower()]
        valid_citations_count = max(0, total_citations - len(invalid_cite_errors))
        citation_validity_rate = round(valid_citations_count / total_citations, 4) if total_citations > 0 else 0.0

        # Citation Precision (citations with non-empty source_id/url matching sources list)
        source_ids_known = {s.source_id for s in sources}
        valid_source_matches = sum(1 for c in citations if c.source_id in source_ids_known or bool(c.url))
        citation_precision = round(valid_source_matches / total_citations, 4) if total_citations > 0 else 0.0

        # Citation Traceability Rate (claims with valid evidence_ids mapped to citations)
        traceable_claims = sum(
            1 for c in claims
            if c.support_status in (SupportStatus.SUPPORTED, SupportStatus.PARTIALLY_SUPPORTED)
            and len(c.evidence_ids) > 0
        )
        citation_traceability_rate = round(traceable_claims / total_claims, 4) if total_claims > 0 else 0.0

        # Unique cited documents and sources
        unique_docs = len({s.source_id for s in sources if s.source_id})
        unique_srcs = len({s.url or s.source_id for s in sources if s.url or s.source_id})

        # Evidence Dimension Coverage
        exp_dims = expected_dimensions or []
        if exp_dims:
            # Combine all summary, claim text, and citation titles
            full_answer_text = (
                f"{answer.summary} "
                + " ".join(c.text for c in claims)
                + " ".join(cite.title for cite in citations)
            ).lower()

            covered_dims = 0
            for dim in exp_dims:
                dim_words = [w for w in re.split(r"\s+", dim.lower()) if len(w) > 2]
                if any(w in full_answer_text for w in dim_words):
                    covered_dims += 1
            dimension_coverage = round(covered_dims / len(exp_dims), 4)
        else:
            dimension_coverage = 1.0

        details = {
            "verif_valid": verif.valid,
            "verif_errors": verif.errors,
            "verif_warnings": verif.warnings,
            "unsupported_claim_ids": verif.unsupported_claim_ids,
            "retrieval_telemetry": answer.metadata.retrieval if hasattr(answer.metadata, "retrieval") else None,
        }

        return AnswerEvaluationScore(
            query_id=query_id or answer.question[:10],
            question=answer.question,
            has_claims=has_claims,
            total_claims=total_claims,
            supported_claims=supported_count,
            partially_supported_claims=part_count,
            unsupported_claims=unsupp_count,
            claim_support_rate=claim_support_rate,
            unsupported_claim_rate=unsupported_claim_rate,
            total_citations=total_citations,
            citation_coverage=citation_coverage,
            citation_validity_rate=citation_validity_rate,
            citation_precision=citation_precision,
            citation_traceability_rate=citation_traceability_rate,
            unique_cited_documents=unique_docs,
            unique_cited_sources=unique_srcs,
            evidence_dimension_coverage=dimension_coverage,
            insufficient_evidence=answer.insufficient_evidence,
            details=details,
        )

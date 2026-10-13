"""
Citation and Grounding Verification Engine.
Performs deterministic audits of LLM claims against the retrieved EvidencePack.
"""
from __future__ import annotations

import time
import structlog

from ranking.evidence import EvidencePack, EvidencePassage
from synthesis.models import (
    Citation,
    CitationVerificationResult,
    ClaimItem,
    SupportStatus,
)

logger = structlog.get_logger(__name__)


class CitationVerifier:
    """
    Verifies that all synthesized claims and citations are strictly grounded in retrieved evidence.
    """

    @classmethod
    def verify(
        cls,
        claims: list[ClaimItem],
        citations: list[Citation],
        id_to_passage: dict[str, EvidencePassage],
        evidence_pack: EvidencePack,
    ) -> tuple[CitationVerificationResult, float]:
        """
        Audit claims, citations, and evidence bindings.
        
        Returns:
            Tuple of (CitationVerificationResult, latency_ms)
        """
        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []
        unsupported_claim_ids: list[str] = []
        verified_claim_count = 0

        known_evidence_ids = set(id_to_passage.keys())
        known_citation_ids = {c.citation_id: c for c in citations}
        used_citation_ids: set[str] = set()

        # Check 6: Empty Evidence Check
        if not evidence_pack.selected_passages or not id_to_passage:
            errors.append("EvidencePack contains zero passages; cannot ground factual claims.")

        for claim in claims:
            # Check 3: Claim Grounding Check (Must have evidence)
            if not claim.evidence_ids:
                claim.support_status = SupportStatus.UNSUPPORTED
                unsupported_claim_ids.append(claim.claim_id)
                errors.append(f"Claim '{claim.claim_id}' has no supporting evidence references.")
                continue

            # Check 1: Evidence Existence Check
            valid_evidence_refs = []
            invalid_evidence_refs = []
            for ev_id in claim.evidence_ids:
                if ev_id in known_evidence_ids:
                    valid_evidence_refs.append(ev_id)
                else:
                    invalid_evidence_refs.append(ev_id)
                    errors.append(
                        f"Claim '{claim.claim_id}' cites non-existent evidence ID '{ev_id}' (hallucinated reference)."
                    )

            # Check 2: Citation Existence Check
            for cite_id in claim.citation_ids:
                if cite_id in known_citation_ids:
                    used_citation_ids.add(cite_id)
                else:
                    warnings.append(f"Claim '{claim.claim_id}' references unknown citation '{cite_id}'.")

            # Check 5: Source Consistency Check
            for ev_id in valid_evidence_refs:
                passage = id_to_passage[ev_id]
                if not passage.document_id or not passage.source_id:
                    warnings.append(f"Evidence '{ev_id}' is missing required provenance identifiers.")

            # Assign Deterministic Support Status
            if not valid_evidence_refs:
                claim.support_status = SupportStatus.UNSUPPORTED
                unsupported_claim_ids.append(claim.claim_id)
            elif invalid_evidence_refs:
                claim.support_status = SupportStatus.PARTIALLY_SUPPORTED
            else:
                claim.support_status = SupportStatus.SUPPORTED
                verified_claim_count += 1

        # Check 4: No Orphan Citations Check
        for cite_id in known_citation_ids:
            if cite_id not in used_citation_ids:
                warnings.append(f"Citation '{cite_id}' is defined but not cited by any claim (orphan citation).")

        total_claims = len(claims)
        grounding_rate = (verified_claim_count / total_claims) if total_claims > 0 else (1.0 if not errors else 0.0)
        is_valid = (len(errors) == 0) and (total_claims == 0 or verified_claim_count == total_claims)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        result = CitationVerificationResult(
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            unsupported_claim_ids=unsupported_claim_ids,
            verified_claim_count=verified_claim_count,
            total_claim_count=total_claims,
            grounding_rate=round(grounding_rate, 4),
        )

        logger.info(
            "citation_verification_completed",
            valid=is_valid,
            verified_claims=verified_claim_count,
            total_claims=total_claims,
            grounding_rate=result.grounding_rate,
            errors_count=len(errors),
            latency_ms=latency_ms,
        )

        return result, latency_ms

"""
Research Answer Service orchestrating Query Planning, Grounded Synthesis,
Citation Mapping, and Deterministic Verification.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Literal
from pydantic import BaseModel, Field
import structlog

from core.config import Settings, get_settings
from llm.base import LLMProvider
from llm.factory import get_llm_provider
from llm.models import LLMGenerationError
from ranking.evidence import EvidencePack
from query.models import MergedEvidencePack
from query.planner import (
    PlanningLatencyProfile,
    ResearchPlanRetrievalRequest,
    ResearchPlanningService,
)
from services.research_retrieval_service import (
    ResearchRetrievalRequest,
    ResearchRetrievalService,
)
from synthesis.citation_mapper import CitationMapper
from synthesis.citation_verifier import CitationVerifier
from synthesis.models import (
    AnswerEvidence,
    ClaimItem,
    ResearchAnswer,
    ResearchAnswerSection,
    SupportStatus,
    SynthesisMetadata,
)
from synthesis.prompt_builder import GroundedPromptBuilder

logger = structlog.get_logger(__name__)


class ResearchAnswerRequest(BaseModel):
    """Request schema for generating a grounded research answer."""
    query: str = Field(..., min_length=1, description="User research question")
    enable_decomposition: bool = Field(default=True, description="Enable query intelligence and decomposition")
    decomposition_mode: Literal["auto", "llm", "deterministic"] = Field(
        default="auto",
        description="Decomposition execution strategy",
    )
    max_subqueries: int = Field(default=5, ge=1, le=10, description="Maximum subqueries to generate")
    candidate_k: int = Field(default=25, ge=1, le=100, description="Candidate retrieval pool size")
    evidence_k: int = Field(default=6, ge=1, le=20, description="Target evidence passages")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Generation temperature")
    max_tokens: int | None = Field(default=2048, ge=1, le=8192, description="Max generation tokens")
    provider_override: str | None = Field(default=None, description="Optional LLM provider override")


def clean_json_text(raw_text: str) -> str:
    """Strip markdown codeblock wrappers if present in model output."""
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


class ResearchAnswerService:
    """
    Coordinates end-to-end research answer generation:
    Query Intelligence -> Planning -> Parallel Retrieval -> Evidence Merging
    -> LLM Synthesis -> Citation Mapping -> Verification
    """

    def __init__(
        self,
        retrieval_service: ResearchRetrievalService | None = None,
        planning_service: ResearchPlanningService | None = None,
        llm_provider: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrieval_service = retrieval_service or ResearchRetrievalService(settings=self.settings)
        self.llm_provider = llm_provider or get_llm_provider(settings=self.settings)
        self.planning_service = planning_service or ResearchPlanningService(
            retrieval_service=self.retrieval_service,
            llm_provider=self.llm_provider,
            settings=self.settings,
        )

    async def answer_question(
        self,
        request: ResearchAnswerRequest,
    ) -> ResearchAnswer:
        """
        Execute end-to-end grounded research synthesis.
        """
        total_start = time.perf_counter()

        query_analysis_dict: dict[str, Any] | None = None
        research_plan_dict: dict[str, Any] | None = None
        retrieval_metrics_dict: dict[str, Any] | None = None
        planning_lat = 0.0

        # ── Step 1: Retrieve and Package Evidence ─────────────────────────────
        if request.enable_decomposition:
            plan_req = ResearchPlanRetrievalRequest(
                query=request.query,
                enable_decomposition=True,
                decomposition_mode=request.decomposition_mode,
                candidate_k=request.candidate_k,
                evidence_k=request.evidence_k,
                max_subqueries=request.max_subqueries,
            )
            merged_pack, plan_lat_profile, plan = await self.planning_service.plan_and_retrieve(plan_req)
            evidence_pack: EvidencePack = merged_pack
            retrieval_lat = plan_lat_profile.subquery_retrieval_wall_clock_ms
            planning_lat = plan_lat_profile.total_query_planning_latency_ms

            query_analysis_dict = plan.analysis.model_dump()
            research_plan_dict = {
                "subquery_count": len(plan.subqueries),
                "strategy": plan.strategy,
                "subqueries": [sq.model_dump() for sq in plan.subqueries],
            }
            retrieval_metrics_dict = merged_pack.coverage_metrics.model_dump()
        else:
            retrieval_req = ResearchRetrievalRequest(
                query=request.query,
                candidate_k=request.candidate_k,
                evidence_k=request.evidence_k,
            )
            retrieval_resp = await self.retrieval_service.retrieve_evidence(retrieval_req)
            evidence_pack = retrieval_resp.evidence_pack
            retrieval_lat = retrieval_resp.latency_breakdown.total_retrieval_latency_ms

        # ── Step 2: Handle Insufficient Evidence Case ────────────────────────
        if not evidence_pack.selected_passages:
            verif_res, _ = CitationVerifier.verify(
                claims=[],
                citations=[],
                id_to_passage={},
                evidence_pack=evidence_pack,
            )
            total_lat = round((time.perf_counter() - total_start) * 1000, 2)
            meta = SynthesisMetadata(
                provider=self.llm_provider.provider_name,
                model=self.llm_provider.model_name,
                planning_latency_ms=planning_lat,
                retrieval_latency_ms=retrieval_lat,
                generation_latency_ms=0.0,
                verification_latency_ms=0.0,
                total_latency_ms=total_lat,
                query_analysis=query_analysis_dict,
                research_plan=research_plan_dict,
                retrieval=retrieval_metrics_dict,
            )
            return ResearchAnswer(
                question=request.query,
                summary="Insufficient evidence to answer this reliably from the retrieved sources.",
                sections=[],
                claims=[],
                citations=[],
                sources=[],
                limitations=["No relevant research passages were found in the index for this query."],
                verification=verif_res,
                metadata=meta,
                insufficient_evidence=True,
                query_analysis=query_analysis_dict,
                research_plan=research_plan_dict,
            )

        # ── Step 3: Build Grounded Prompt ─────────────────────────────────────
        _, id_to_passage = GroundedPromptBuilder.format_evidence_block(evidence_pack)
        user_prompt = GroundedPromptBuilder.build_user_prompt(request.query, evidence_pack)

        # ── Step 4: LLM Generation ───────────────────────────────────────────
        active_provider = (
            get_llm_provider(request.provider_override, settings=self.settings)
            if request.provider_override
            else self.llm_provider
        )

        llm_resp = await active_provider.generate(
            prompt=user_prompt,
            system_prompt=GroundedPromptBuilder.SYSTEM_PROMPT,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_format="json",
        )
        gen_lat = llm_resp.latency_ms

        # ── Step 5: Parse Structured JSON Response ────────────────────────────
        raw_content = clean_json_text(llm_resp.content)
        # A provider failure (truncated / malformed structured output) must surface as an
        # error. It is NOT "insufficient evidence", and no placeholder answer is fabricated.
        try:
            parsed_data = json.loads(raw_content)
            if not isinstance(parsed_data, dict):
                raise ValueError("top-level JSON value is not an object")
        except Exception as e:
            logger.error(
                "llm_json_parse_failed",
                error_type=type(e).__name__,
                finish_reason=str(llm_resp.finish_reason),
                output_chars=len(raw_content),
            )
            raise LLMGenerationError(
                "Structured output could not be parsed "
                f"(finish_reason={llm_resp.finish_reason}, {len(raw_content)} chars): "
                "the response was truncated or malformed."
            ) from e

        summary_text = parsed_data.get("summary", "")
        insufficient_flag = parsed_data.get("insufficient_evidence", False)
        limitations = parsed_data.get("limitations", [])

        raw_claims = parsed_data.get("claims", [])
        claims: list[ClaimItem] = []
        for c in raw_claims:
            if isinstance(c, dict) and c.get("text"):
                claim_id = c.get("claim_id") or f"claim_{len(claims)+1:03d}"
                ev_ids = c.get("evidence_ids") or []
                claims.append(ClaimItem(claim_id=claim_id, text=c["text"], evidence_ids=ev_ids))

        # Reconstruct sections. Every section claim is the SAME verified claim object as the
        # top-level claim with identical text (or is appended to `claims` if it only appears
        # in a section), so citation mapping and verification cover everything displayed.
        def _norm(t: str) -> str:
            return " ".join(t.split()).lower()

        claims_by_text: dict[str, ClaimItem] = {}
        for c in claims:
            claims_by_text.setdefault(_norm(c.text), c)
        used_ids = {c.claim_id for c in claims}

        section_specs: list[tuple[str, list[ClaimItem]]] = []
        for s_raw in parsed_data.get("sections", []) or []:
            if not (isinstance(s_raw, dict) and s_raw.get("title")):
                continue
            members: list[ClaimItem] = []
            for sc in s_raw.get("claims", []) or []:
                if not (isinstance(sc, dict) and sc.get("text")):
                    continue
                key = _norm(sc["text"])
                shared = claims_by_text.get(key)
                sc_ev = sc.get("evidence_ids") or []
                if shared is None:
                    cid = sc.get("claim_id") or f"claim_{len(claims) + 1:03d}"
                    n = len(claims) + 1
                    while cid in used_ids:
                        n += 1
                        cid = f"claim_{n:03d}"
                    shared = ClaimItem(claim_id=cid, text=sc["text"], evidence_ids=list(sc_ev))
                    claims.append(shared)
                    claims_by_text[key] = shared
                    used_ids.add(cid)
                else:
                    for ev in sc_ev:
                        if ev not in shared.evidence_ids:
                            shared.evidence_ids.append(ev)
                members.append(shared)
            section_specs.append((s_raw["title"], members))

        # ── Step 6: Map Citations & Provenance ────────────────────────────────
        citations, sources = CitationMapper.map_citations(claims, id_to_passage)

        # ── Step 7: Verify Grounding and Attribution ──────────────────────────
        verif_result, verif_lat = CitationVerifier.verify(
            claims=claims,
            citations=citations,
            id_to_passage=id_to_passage,
            evidence_pack=evidence_pack,
        )

        # Sections are materialised AFTER verification so they carry the verified status
        # and citation ids of the shared claim objects.
        sections: list[ResearchAnswerSection] = [
            ResearchAnswerSection(title=title, claims=[m.model_copy(deep=True) for m in members])
            for title, members in section_specs
        ]

        total_lat = round((time.perf_counter() - total_start) * 1000, 2)

        # Count claim statuses
        supp_count = sum(1 for c in claims if c.support_status == SupportStatus.SUPPORTED)
        part_count = sum(1 for c in claims if c.support_status == SupportStatus.PARTIALLY_SUPPORTED)
        unsupp_count = sum(1 for c in claims if c.support_status == SupportStatus.UNSUPPORTED)

        tokens = llm_resp.usage
        metadata = SynthesisMetadata(
            provider=active_provider.provider_name,
            model=active_provider.model_name,
            planning_latency_ms=planning_lat,
            retrieval_latency_ms=retrieval_lat,
            generation_latency_ms=gen_lat,
            verification_latency_ms=verif_lat,
            total_latency_ms=total_lat,
            input_tokens=tokens.input_tokens if tokens else None,
            output_tokens=tokens.output_tokens if tokens else None,
            total_tokens=tokens.total_tokens if tokens else None,
            claim_count=len(claims),
            supported_claim_count=supp_count,
            partially_supported_claim_count=part_count,
            unsupported_claim_count=unsupp_count,
            query_analysis=query_analysis_dict,
            research_plan=research_plan_dict,
            retrieval=retrieval_metrics_dict,
        )

        logger.info(
            "research_answer_completed",
            query=request.query,
            claims=len(claims),
            citations=len(citations),
            grounding_rate=verif_result.grounding_rate,
            total_latency_ms=total_lat,
        )

        evidence_items = [
            AnswerEvidence(
                evidence_id=ev_id,
                passage_id=p.passage_id,
                document_id=p.document_id,
                title=p.title,
                source_type=p.source_type,
                source_id=p.source_id,
                url=p.url,
                text=p.text,
                relevance_score=p.relevance_score,
                rank=p.rank,
                chunk_index=p.chunk_index,
                authors=[str(a) for a in (p.metadata.get("authors") or [])],
                publication_date=(
                    str(p.metadata.get("publication_date") or p.metadata.get("year"))
                    if (p.metadata.get("publication_date") or p.metadata.get("year"))
                    else None
                ),
            )
            for ev_id, p in id_to_passage.items()
        ]

        return ResearchAnswer(
            question=request.query,
            summary=summary_text,
            sections=sections,
            claims=claims,
            citations=citations,
            sources=sources,
            evidence=evidence_items,
            limitations=limitations,
            verification=verif_result,
            metadata=metadata,
            insufficient_evidence=insufficient_flag,
            query_analysis=query_analysis_dict,
            research_plan=research_plan_dict,
        )

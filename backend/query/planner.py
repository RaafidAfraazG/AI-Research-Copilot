"""
Research Planning and Retrieval Service.
Orchestrates Query Intelligence, Query Decomposition, Parallel Subquery Retrieval,
and Unified Evidence Merging.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Literal
from pydantic import BaseModel, Field
import structlog

from core.config import Settings, get_settings
from core.exceptions import AppException
from llm.base import LLMProvider
from llm.factory import get_llm_provider
from query.complexity import QueryComplexityAnalyzer
from query.decomposer import QueryDecomposer
from query.intent import QueryIntentClassifier
from query.merger import EvidenceMerger
from query.models import (
    MergedEvidencePack,
    QueryAnalysis,
    QueryComplexity,
    QueryIntent,
    ResearchPlan,
    ResearchSubQuery,
)
from services.research_retrieval_service import (
    ResearchRetrievalRequest,
    ResearchRetrievalResponse,
    ResearchRetrievalService,
)

logger = structlog.get_logger(__name__)


class ResearchPlanRetrievalRequest(BaseModel):
    """
    Request configuration for query planning and parallel retrieval.
    """
    query: str = Field(..., min_length=1, description="User research question")
    enable_decomposition: bool = Field(default=True, description="Enable query intelligence and decomposition")
    decomposition_mode: Literal["auto", "llm", "deterministic"] = Field(
        default="auto",
        description="Decomposition execution strategy",
    )
    candidate_k: int = Field(default=25, ge=1, le=100, description="Total candidate pool size per retrieval")
    evidence_k: int = Field(default=6, ge=1, le=20, description="Target final evidence passage count")
    max_subqueries: int = Field(default=5, ge=1, le=10, description="Maximum subqueries to generate")
    max_evidence_per_subquery: int = Field(default=4, ge=1, le=10, description="Max evidence items per subquery")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Hybrid retrieval balance")
    source_types: list[str] | None = Field(default=None, description="Filter by source types")
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters")


class PlanningLatencyProfile(BaseModel):
    """Latency metrics for query planning, parallel retrieval, and evidence merging."""
    query_analysis_latency_ms: float = 0.0
    decomposition_latency_ms: float = 0.0
    subquery_retrieval_wall_clock_ms: float = 0.0
    sum_of_subquery_latencies_ms: float = 0.0
    evidence_merge_latency_ms: float = 0.0
    total_query_planning_latency_ms: float = 0.0


class ResearchPlanningService:
    """
    High-level coordinator uniting:
    QueryAnalysis -> QueryDecomposer -> Parallel Retrieval -> EvidenceMerger -> MergedEvidencePack
    """

    def __init__(
        self,
        retrieval_service: ResearchRetrievalService | None = None,
        llm_provider: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrieval_service = retrieval_service or ResearchRetrievalService(settings=self.settings)
        self.llm_provider = llm_provider or get_llm_provider(settings=self.settings)
        self.decomposer = QueryDecomposer(
            llm_provider=self.llm_provider,
            max_subqueries=self.settings.max_subqueries if hasattr(self.settings, "max_subqueries") else 5,
        )
        self.merger = EvidenceMerger(
            max_total_evidence=self.settings.max_total_evidence if hasattr(self.settings, "max_total_evidence") else 12,
            max_evidence_per_subquery=self.settings.max_evidence_per_subquery if hasattr(self.settings, "max_evidence_per_subquery") else 4,
        )

    async def analyze_and_plan(
        self,
        query: str,
        mode: Literal["auto", "llm", "deterministic"] = "auto",
    ) -> tuple[QueryAnalysis, ResearchPlan, PlanningLatencyProfile]:
        """
        Execute query intelligence and plan formulation without running retrieval.
        Useful for planning preview, inspection, and the `/research/plan` endpoint.
        """
        total_start = time.perf_counter()

        # Step 1: Query Analysis
        analysis_start = time.perf_counter()
        analysis = self.decomposer.analyze_query(query)
        analysis_lat = round((time.perf_counter() - analysis_start) * 1000, 2)

        # Step 2: Decomposition
        decomp_start = time.perf_counter()
        plan = await self.decomposer.decompose(query, analysis=analysis, mode=mode)
        decomp_lat = round((time.perf_counter() - decomp_start) * 1000, 2)

        total_lat = round((time.perf_counter() - total_start) * 1000, 2)

        latency_profile = PlanningLatencyProfile(
            query_analysis_latency_ms=analysis_lat,
            decomposition_latency_ms=decomp_lat,
            total_query_planning_latency_ms=total_lat,
        )

        return analysis, plan, latency_profile

    async def plan_and_retrieve(
        self,
        request: ResearchPlanRetrievalRequest,
    ) -> tuple[MergedEvidencePack, PlanningLatencyProfile, ResearchPlan]:
        """
        Execute end-to-end query planning, parallel subquery retrieval, and evidence merging.
        """
        total_start = time.perf_counter()

        # Step 1 & 2: Analyze & Formulate Research Plan
        analysis, plan, plan_lat = await self.analyze_and_plan(
            query=request.query,
            mode=request.decomposition_mode,
        )

        # Optimization: If decomposition is disabled by request or query is simple (single pass)
        if not request.enable_decomposition or not analysis.requires_decomposition or len(plan.subqueries) <= 1:
            logger.info(
                "executing_single_pass_retrieval",
                query=request.query,
                enable_decomposition=request.enable_decomposition,
                requires_decomposition=analysis.requires_decomposition,
            )
            retrieval_start = time.perf_counter()
            single_req = ResearchRetrievalRequest(
                query=request.query,
                candidate_k=request.candidate_k,
                evidence_k=request.evidence_k,
                alpha=request.alpha,
                source_types=request.source_types,
                filters=request.filters,
            )
            retrieval_resp = await self.retrieval_service.retrieve_evidence(single_req)
            retrieval_lat = round((time.perf_counter() - retrieval_start) * 1000, 2)

            # Ensure single subquery in plan
            single_sq = (
                plan.subqueries[0]
                if plan.subqueries
                else ResearchSubQuery(
                    subquery_id="sq_001",
                    text=request.query,
                    purpose="Direct search",
                    intent=analysis.intent,
                    priority=1,
                )
            )
            single_plan = ResearchPlan(
                original_query=request.query,
                analysis=analysis,
                subqueries=[single_sq],
                strategy="single_pass",
                execution_order=[[single_sq.subquery_id]],
            )

            # Merge single result
            merged_pack = self.merger.merge(
                plan=single_plan,
                subquery_results=[(single_sq, retrieval_resp)],
                planning_latency_ms=plan_lat.total_query_planning_latency_ms,
            )

            total_lat = round((time.perf_counter() - total_start) * 1000, 2)
            profile = PlanningLatencyProfile(
                query_analysis_latency_ms=plan_lat.query_analysis_latency_ms,
                decomposition_latency_ms=plan_lat.decomposition_latency_ms,
                subquery_retrieval_wall_clock_ms=retrieval_lat,
                sum_of_subquery_latencies_ms=retrieval_lat,
                evidence_merge_latency_ms=0.0,
                total_query_planning_latency_ms=total_lat,
            )
            return merged_pack, profile, single_plan

        # Step 3: Parallel Subquery Retrieval
        logger.info(
            "executing_parallel_subquery_retrieval",
            query=request.query,
            subqueries=len(plan.subqueries),
            strategy=plan.strategy,
        )

        subquery_retrieval_start = time.perf_counter()

        async def _retrieve_single_subquery(
            sq: ResearchSubQuery,
        ) -> tuple[ResearchSubQuery, ResearchRetrievalResponse | None, float]:
            """Retrieve evidence for a single subquery with error isolation."""
            sq_start = time.perf_counter()
            try:
                sub_req = ResearchRetrievalRequest(
                    query=sq.text,
                    candidate_k=max(15, request.candidate_k // max(1, len(plan.subqueries))),
                    evidence_k=request.max_evidence_per_subquery,
                    alpha=request.alpha,
                    source_types=request.source_types,
                    filters=request.filters,
                )
                resp = await self.retrieval_service.retrieve_evidence(sub_req)
                lat = round((time.perf_counter() - sq_start) * 1000, 2)
                return sq, resp, lat
            except AppException:
                # Index-integrity / embedding errors must fail the whole request,
                # not be downgraded to a silently "failed" subquery.
                raise
            except Exception as e:
                logger.error("subquery_retrieval_failed", subquery_id=sq.subquery_id, query=sq.text, error=str(e))
                lat = round((time.perf_counter() - sq_start) * 1000, 2)
                return sq, None, lat

        # Execute parallel retrieval across all planned subqueries
        tasks = [_retrieve_single_subquery(sq) for sq in plan.subqueries]
        results = await asyncio.gather(*tasks)

        subquery_wall_clock_ms = round((time.perf_counter() - subquery_retrieval_start) * 1000, 2)
        sum_of_latencies_ms = sum(r[2] for r in results)

        subquery_pairs: list[tuple[ResearchSubQuery, ResearchRetrievalResponse | None]] = [
            (sq, resp) for sq, resp, _ in results
        ]
        failed_subqueries = [sq.subquery_id for sq, resp, _ in results if resp is None]

        # Step 4: Evidence Merging and Deduplication
        merge_start = time.perf_counter()
        merged_pack = self.merger.merge(
            plan=plan,
            subquery_results=subquery_pairs,
            failed_subquery_ids=failed_subqueries,
            planning_latency_ms=plan_lat.total_query_planning_latency_ms,
        )
        merge_lat = round((time.perf_counter() - merge_start) * 1000, 2)

        total_lat = round((time.perf_counter() - total_start) * 1000, 2)

        latency_profile = PlanningLatencyProfile(
            query_analysis_latency_ms=plan_lat.query_analysis_latency_ms,
            decomposition_latency_ms=plan_lat.decomposition_latency_ms,
            subquery_retrieval_wall_clock_ms=subquery_wall_clock_ms,
            sum_of_subquery_latencies_ms=sum_of_latencies_ms,
            evidence_merge_latency_ms=merge_lat,
            total_query_planning_latency_ms=total_lat,
        )

        return merged_pack, latency_profile, plan

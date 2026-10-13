"""
Research Planning, Retrieval, and Evidence API endpoints.
Provides query intelligence, subquery decomposition, parallel retrieval,
cross-encoder reranking, and grounded LLM answer synthesis.
"""
from __future__ import annotations

from typing import Literal
from fastapi import APIRouter

from api.deps import get_answer_service, get_planning_service, get_retrieval_service
from pydantic import BaseModel, Field

from query.models import QueryAnalysis, ResearchPlan
from query.planner import PlanningLatencyProfile
from services.research_retrieval_service import (
    ResearchRetrievalRequest,
    ResearchRetrievalResponse,
)
from synthesis.answer_service import ResearchAnswerRequest
from synthesis.models import ResearchAnswer

router = APIRouter(prefix="/research", tags=["research"])


class ResearchPlanRequest(BaseModel):
    """Request schema for query analysis and research decomposition without retrieval."""
    query: str = Field(..., min_length=1, description="User research question to analyze and plan")
    decomposition_mode: Literal["auto", "llm", "deterministic"] = Field(
        default="auto",
        description="Decomposition execution strategy",
    )


class ResearchPlanResponse(BaseModel):
    """Structured response containing query analysis, decomposed research plan, and latencies."""
    query: str
    analysis: QueryAnalysis
    plan: ResearchPlan
    latency_profile: PlanningLatencyProfile


@router.post(
    "/plan",
    response_model=ResearchPlanResponse,
    summary="Analyze query and generate decomposed research plan",
    description=(
        "Performs deterministic complexity detection and intent classification, extracts "
        "entities/constraints, and decomposes the research question into targeted subqueries "
        "without executing retrieval."
    ),
)
async def generate_research_plan(
    request: ResearchPlanRequest,
) -> ResearchPlanResponse:
    """Generate QueryAnalysis and ResearchPlan for the given research query."""
    planner = get_planning_service()
    analysis, plan, lat = await planner.analyze_and_plan(
        query=request.query,
        mode=request.decomposition_mode,
    )
    return ResearchPlanResponse(
        query=request.query,
        analysis=analysis,
        plan=plan,
        latency_profile=lat,
    )


@router.post(
    "/retrieve",
    response_model=ResearchRetrievalResponse,
    summary="Retrieve and rerank research evidence",
    description=(
        "Executes hybrid candidate retrieval (BM25 + dense vector), applies cross-encoder "
        "reranking, filters near-duplicates, applies document diversity controls, "
        "and returns an assembled EvidencePack with latency profiling."
    ),
)
async def retrieve_research_evidence(
    request: ResearchRetrievalRequest,
) -> ResearchRetrievalResponse:
    """Execute end-to-end research retrieval and evidence assembly."""
    return await get_retrieval_service().retrieve_evidence(request)


@router.post(
    "/answer",
    response_model=ResearchAnswer,
    summary="Generate grounded research answer with citations",
    description=(
        "Analyzes query intent/complexity, decomposes complex queries into targeted subqueries, "
        "executes parallel hybrid retrieval and cross-encoder reranking, merges and deduplicates evidence, "
        "synthesizes a structured grounded answer with an LLM, verifies claim grounding, "
        "and produces verifiable bibliographic citations."
    ),
)
async def generate_research_answer(
    request: ResearchAnswerRequest,
) -> ResearchAnswer:
    """Generate grounded research answer strictly bound to retrieved evidence."""
    return await get_answer_service().answer_question(request)

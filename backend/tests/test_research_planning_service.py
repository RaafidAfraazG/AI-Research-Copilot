"""
Unit tests for ResearchPlanningService (planning, parallel retrieval orchestration, and error resilience).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from ranking.evidence import EvidencePack, EvidencePassage
from query.models import QueryComplexity, QueryIntent
from query.planner import ResearchPlanRetrievalRequest, ResearchPlanningService
from services.research_retrieval_service import LatencyBreakdown, ResearchRetrievalRequest, ResearchRetrievalResponse


def _mock_retrieval_response(query: str) -> ResearchRetrievalResponse:
    p1 = EvidencePassage(
        passage_id=f"pass_{hash(query) % 1000:04d}",
        document_id=f"doc_{hash(query) % 100:03d}",
        text=f"Evidence text for query: {query}",
        title=f"Title for {query}",
        source_type="arxiv",
        source_id="2301.0001",
        url="https://arxiv.org/abs/2301.0001",
        relevance_score=0.92,
        rank=1,
    )
    pack = EvidencePack(
        query=query,
        selected_passages=[p1],
        total_candidates=10,
        total_evidence=1,
    )
    return ResearchRetrievalResponse(
        query=query,
        total_candidates=10,
        total_evidence=1,
        candidates=[],
        reranked=[],
        evidence_pack=pack,
        latency_breakdown=LatencyBreakdown(total_retrieval_latency_ms=15.0),
    )


class TestResearchPlanningService:
    """Test suite verifying end-to-end planning and retrieval orchestration."""

    @pytest.mark.asyncio
    async def test_analyze_and_plan_dedicated(self) -> None:
        planner = ResearchPlanningService()
        query = "Compare dense retrieval and BM25 for RAG systems"

        analysis, plan, lat = await planner.analyze_and_plan(query, mode="deterministic")

        assert analysis.intent == QueryIntent.COMPARATIVE
        assert analysis.complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX)
        assert analysis.requires_decomposition is True
        assert len(plan.subqueries) >= 3
        assert lat.total_query_planning_latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_simple_query_executes_single_pass(self) -> None:
        mock_retrieval_svc = MagicMock()
        mock_retrieval_svc.retrieve_evidence = AsyncMock(
            side_effect=lambda req: _mock_retrieval_response(req.query)
        )

        planner = ResearchPlanningService(retrieval_service=mock_retrieval_svc)
        req = ResearchPlanRetrievalRequest(
            query="What is BM25?",
            enable_decomposition=True,
        )

        merged_pack, lat_profile, plan = await planner.plan_and_retrieve(req)

        # Retrieval should have been called exactly once
        assert mock_retrieval_svc.retrieve_evidence.call_count == 1
        assert len(plan.subqueries) == 1
        assert plan.strategy == "single_pass"
        assert len(merged_pack.selected_passages) == 1
        assert merged_pack.coverage_metrics.coverage == 1.0

    @pytest.mark.asyncio
    async def test_complex_query_executes_parallel_subqueries(self) -> None:
        mock_retrieval_svc = MagicMock()
        mock_retrieval_svc.retrieve_evidence = AsyncMock(
            side_effect=lambda req: _mock_retrieval_response(req.query)
        )

        planner = ResearchPlanningService(retrieval_service=mock_retrieval_svc)
        req = ResearchPlanRetrievalRequest(
            query="Compare dense retrieval and BM25 for RAG systems, explain strengths and weaknesses, and identify when each performs better.",
            enable_decomposition=True,
            decomposition_mode="deterministic",
        )

        merged_pack, lat_profile, plan = await planner.plan_and_retrieve(req)

        # Multiple subqueries executed in parallel
        subquery_count = len(plan.subqueries)
        assert subquery_count >= 3
        assert mock_retrieval_svc.retrieve_evidence.call_count == subquery_count
        assert len(merged_pack.selected_passages) >= 1
        assert merged_pack.coverage_metrics.subqueries == subquery_count
        assert merged_pack.coverage_metrics.coverage > 0.0

    @pytest.mark.asyncio
    async def test_disable_decomposition_flag_forces_single_pass(self) -> None:
        mock_retrieval_svc = MagicMock()
        mock_retrieval_svc.retrieve_evidence = AsyncMock(
            side_effect=lambda req: _mock_retrieval_response(req.query)
        )

        planner = ResearchPlanningService(retrieval_service=mock_retrieval_svc)
        req = ResearchPlanRetrievalRequest(
            query="Compare dense retrieval and BM25 for RAG systems",
            enable_decomposition=False,
        )

        merged_pack, lat_profile, plan = await planner.plan_and_retrieve(req)

        assert mock_retrieval_svc.retrieve_evidence.call_count == 1
        assert len(plan.subqueries) == 1

    @pytest.mark.asyncio
    async def test_partial_subquery_failure_resilience(self) -> None:
        # Mock retrieval where 1 subquery fails
        call_count = 0

        async def _failing_retrieval(req: ResearchRetrievalRequest) -> ResearchRetrievalResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Temporary upstream database timeout")
            return _mock_retrieval_response(req.query)

        mock_retrieval_svc = MagicMock()
        mock_retrieval_svc.retrieve_evidence = AsyncMock(side_effect=_failing_retrieval)

        planner = ResearchPlanningService(retrieval_service=mock_retrieval_svc)
        req = ResearchPlanRetrievalRequest(
            query="Compare dense retrieval and BM25 for RAG systems",
            enable_decomposition=True,
            decomposition_mode="deterministic",
        )

        merged_pack, lat_profile, plan = await planner.plan_and_retrieve(req)

        # Should NOT fail the entire request, but record failed subquery
        assert len(merged_pack.selected_passages) >= 1
        assert len(merged_pack.coverage_metrics.failed_subqueries) == 1

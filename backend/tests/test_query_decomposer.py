"""
Unit tests for QueryDecomposer (deterministic and LLM-assisted).
"""
import pytest

from llm.mock_provider import MockLLMProvider
from query.decomposer import QueryDecomposer
from query.models import QueryComplexity, QueryIntent


class TestQueryDecomposer:
    """Test suite verifying subquery formulation, constraint preservation, and safety limits."""

    def test_simple_query_produces_single_subquery(self) -> None:
        decomposer = QueryDecomposer()
        query = "What is BM25?"
        analysis = decomposer.analyze_query(query)
        plan = decomposer.decompose_deterministic(query, analysis)

        assert len(plan.subqueries) == 1
        assert plan.strategy == "single_pass"
        assert plan.subqueries[0].subquery_id == "sq_001"
        assert plan.subqueries[0].text == query
        assert plan.subqueries[0].intent == QueryIntent.FACTUAL

    def test_comparative_query_decomposition(self) -> None:
        decomposer = QueryDecomposer(max_subqueries=5)
        query = (
            "Compare dense retrieval and BM25 for RAG systems, explain their strengths and weaknesses, "
            "and identify when each performs better."
        )
        analysis = decomposer.analyze_query(query)
        plan = decomposer.decompose_deterministic(query, analysis)

        assert len(plan.subqueries) >= 3
        sub_texts = [sq.text.lower() for sq in plan.subqueries]

        # Target A and Target B represented
        assert any("dense retrieval" in t for t in sub_texts)
        assert any("bm25" in t for t in sub_texts)
        assert any("vs" in t or "comparative" in t for t in sub_texts)

        # Every subquery has a purpose and normalized ID
        for idx, sq in enumerate(plan.subqueries, start=1):
            assert sq.subquery_id == f"sq_{idx:03d}"
            assert len(sq.purpose) > 5
            assert sq.priority in (1, 2)

    def test_multi_part_query_decomposition(self) -> None:
        decomposer = QueryDecomposer()
        query = "How does RAG improve question answering and what are its primary limitations and failure modes?"
        analysis = decomposer.analyze_query(query)
        plan = decomposer.decompose_deterministic(query, analysis)

        assert len(plan.subqueries) >= 2
        sub_texts = [sq.text.lower() for sq in plan.subqueries]
        assert any("rag" in t or "improve" in t for t in sub_texts)
        assert any("limitations" in t or "failure" in t for t in sub_texts)

    def test_max_subqueries_limit_enforced(self) -> None:
        decomposer = QueryDecomposer(max_subqueries=3)
        query = "Compare X and Y and Z, explain A, B, C, and evaluate trade-offs in 2024"
        analysis = decomposer.analyze_query(query)
        plan = decomposer.decompose_deterministic(query, analysis)

        assert len(plan.subqueries) <= 3

    @pytest.mark.asyncio
    async def test_llm_assisted_decomposition(self) -> None:
        mock_llm = MockLLMProvider()
        decomposer = QueryDecomposer(llm_provider=mock_llm)
        query = "Compare cross-encoders and bi-encoders for search latency"
        plan = await decomposer.decompose(query, mode="llm")

        assert len(plan.subqueries) >= 2
        assert plan.subqueries[0].subquery_id == "sq_001"
        assert plan.subqueries[1].subquery_id == "sq_002"

    @pytest.mark.asyncio
    async def test_malformed_llm_output_falls_back_gracefully(self) -> None:
        # Mock LLM returning invalid non-JSON output
        broken_llm = MockLLMProvider(custom_response="This is completely broken and not JSON at all!")
        decomposer = QueryDecomposer(llm_provider=broken_llm)
        query = "Compare dense retrieval and BM25 for RAG systems"
        plan = await decomposer.decompose(query, mode="llm")

        assert plan is not None
        assert len(plan.subqueries) >= 2
        assert plan.subqueries[0].subquery_id == "sq_001"

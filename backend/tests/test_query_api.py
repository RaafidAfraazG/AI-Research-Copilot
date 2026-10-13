"""
Unit and integration tests for Query Intelligence API endpoints:
- POST /api/v1/research/plan
- POST /api/v1/research/answer (with decomposition and metadata)
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from main import app

pytestmark = pytest.mark.usefixtures("mock_index")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_api_research_plan_endpoint() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post(
            "/api/v1/research/plan",
            json={
                "query": "Compare dense retrieval and BM25 for RAG systems, explain strengths and weaknesses, and identify when each performs better.",
                "decomposition_mode": "deterministic",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"].startswith("Compare dense retrieval and BM25")
        assert data["analysis"]["intent"] == "COMPARATIVE"
        assert data["analysis"]["requires_decomposition"] is True
        assert len(data["plan"]["subqueries"]) >= 3
        assert data["latency_profile"]["total_query_planning_latency_ms"] >= 0.0


@pytest.mark.asyncio
async def test_api_research_plan_simple_query() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post(
            "/api/v1/research/plan",
            json={
                "query": "What is BM25?",
                "decomposition_mode": "deterministic",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis"]["complexity"] == "SIMPLE"
        assert data["analysis"]["requires_decomposition"] is False
        assert len(data["plan"]["subqueries"]) == 1
        assert data["plan"]["strategy"] == "single_pass"


@pytest.mark.asyncio
async def test_api_research_answer_with_decomposition() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post(
            "/api/v1/research/answer",
            json={
                "query": "Compare dense retrieval and BM25 for RAG systems",
                "enable_decomposition": True,
                "decomposition_mode": "deterministic",
                "candidate_k": 10,
                "evidence_k": 4,
                "provider_override": "mock",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "question" in data
        assert "metadata" in data
        assert "query_analysis" in data
        assert data["query_analysis"]["intent"] == "COMPARATIVE"
        assert "research_plan" in data
        assert "subquery_count" in data["research_plan"]
        assert "retrieval" in data["metadata"]
        assert "coverage" in data["metadata"]["retrieval"]


@pytest.mark.asyncio
async def test_api_research_answer_without_decomposition() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post(
            "/api/v1/research/answer",
            json={
                "query": "What is BM25?",
                "enable_decomposition": False,
                "provider_override": "mock",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "question" in data
        assert "verification" in data


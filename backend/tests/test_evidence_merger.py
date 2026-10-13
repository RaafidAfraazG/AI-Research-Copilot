"""
Unit tests for EvidenceMerger (deduplication, provenance, diversity, and coverage metrics).
"""
import pytest

from ranking.evidence import EvidencePack, EvidencePassage
from query.merger import EvidenceMerger
from query.models import QueryAnalysis, QueryComplexity, QueryIntent, ResearchPlan, ResearchSubQuery
from services.research_retrieval_service import LatencyBreakdown, ResearchRetrievalResponse


def _make_passage(
    passage_id: str,
    doc_id: str,
    text: str,
    score: float,
    source_type: str = "arxiv",
) -> EvidencePassage:
    return EvidencePassage(
        passage_id=passage_id,
        document_id=doc_id,
        text=text,
        title=f"Title for {doc_id}",
        source_type=source_type,
        source_id=f"src_{doc_id}",
        url=f"https://arxiv.org/abs/{doc_id}",
        relevance_score=score,
        rank=1,
        provenance={"source": "test"},
    )


def _make_response(query: str, passages: list[EvidencePassage]) -> ResearchRetrievalResponse:
    pack = EvidencePack(
        query=query,
        selected_passages=passages,
        total_candidates=len(passages) * 2,
        total_evidence=len(passages),
    )
    return ResearchRetrievalResponse(
        query=query,
        total_candidates=len(passages) * 2,
        total_evidence=len(passages),
        candidates=[],
        reranked=[],
        evidence_pack=pack,
        latency_breakdown=LatencyBreakdown(total_retrieval_latency_ms=10.0),
    )


class TestEvidenceMerger:
    """Test suite verifying multi-subquery evidence merging and deduplication."""

    def test_deduplication_exact_passage_id(self) -> None:
        merger = EvidenceMerger(max_total_evidence=10, max_evidence_per_subquery=4)

        sq1 = ResearchSubQuery(subquery_id="sq_001", text="dense retrieval", purpose="P1", priority=1)
        sq2 = ResearchSubQuery(subquery_id="sq_002", text="BM25 search", purpose="P2", priority=1)

        p_shared = _make_passage("pass_shared", "doc_1", "Shared passage text on hybrid retrieval.", 0.95)
        p_unique1 = _make_passage("pass_u1", "doc_2", "Unique passage 1 on dense vectors.", 0.85)
        p_unique2 = _make_passage("pass_u2", "doc_3", "Unique passage 2 on lexical BM25.", 0.80)

        resp1 = _make_response("dense retrieval", [p_shared, p_unique1])
        resp2 = _make_response("BM25 search", [p_shared, p_unique2])

        plan = ResearchPlan(
            original_query="Compare dense and BM25",
            analysis=QueryAnalysis(
                original_query="Compare dense and BM25",
                intent=QueryIntent.COMPARATIVE,
                complexity=QueryComplexity.MODERATE,
                requires_decomposition=True,
            ),
            subqueries=[sq1, sq2],
        )

        merged = merger.merge(plan, [(sq1, resp1), (sq2, resp2)])

        # Shared passage should only appear once
        passage_ids = [p.passage_id for p in merged.selected_passages]
        assert len(passage_ids) == 3
        assert passage_ids.count("pass_shared") == 1

        # Shared passage provenance tracks both subqueries
        shared_entry = next(p for p in merged.selected_passages if p.passage_id == "pass_shared")
        assert "sq_001" in shared_entry.provenance.get("subquery_ids", [])
        assert "sq_002" in shared_entry.provenance.get("subquery_ids", [])

        # Subquery evidence map maps shared passage to both
        assert "pass_shared" in merged.subquery_evidence_map["sq_001"]
        assert "pass_shared" in merged.subquery_evidence_map["sq_002"]

    def test_near_duplicate_text_suppression(self) -> None:
        merger = EvidenceMerger(similarity_threshold=0.80)

        sq1 = ResearchSubQuery(subquery_id="sq_001", text="query 1", purpose="P1", priority=1)
        sq2 = ResearchSubQuery(subquery_id="sq_002", text="query 2", purpose="P2", priority=1)

        p1 = _make_passage("pass_1", "doc_1", "Dense retrieval improves semantic search recall significantly in modern benchmarks.", 0.90)
        p2 = _make_passage("pass_2", "doc_2", "Dense retrieval improves semantic search recall significantly in modern benchmarks today.", 0.88)

        resp1 = _make_response("q1", [p1])
        resp2 = _make_response("q2", [p2])


        plan = ResearchPlan(
            original_query="original query",
            analysis=QueryAnalysis(original_query="original query"),
            subqueries=[sq1, sq2],
        )

        merged = merger.merge(plan, [(sq1, resp1), (sq2, resp2)])

        # Only one of the near-duplicates should be kept
        assert len(merged.selected_passages) == 1
        assert merged.selected_passages[0].passage_id == "pass_1"

    def test_coverage_metrics_calculation(self) -> None:
        merger = EvidenceMerger()

        sq1 = ResearchSubQuery(subquery_id="sq_001", text="q1", purpose="P1", priority=1)
        sq2 = ResearchSubQuery(subquery_id="sq_002", text="q2", purpose="P2", priority=1)
        sq3 = ResearchSubQuery(subquery_id="sq_003", text="q3", purpose="P3", priority=2)

        p1 = _make_passage("pass_1", "doc_1", "Text 1", 0.9)
        p2 = _make_passage("pass_2", "doc_2", "Text 2", 0.8)

        resp1 = _make_response("q1", [p1])
        resp2 = _make_response("q2", [p2])
        resp3 = _make_response("q3", [])  # Empty evidence for sq3

        plan = ResearchPlan(
            original_query="original",
            analysis=QueryAnalysis(original_query="original"),
            subqueries=[sq1, sq2, sq3],
        )

        merged = merger.merge(plan, [(sq1, resp1), (sq2, resp2), (sq3, resp3)])

        cov = merged.coverage_metrics
        assert cov.subqueries == 3
        assert cov.subqueries_with_evidence == 2
        assert round(cov.coverage, 2) == 0.67
        assert cov.unique_sources == 2
        assert cov.unique_documents == 2

    def test_evidence_budget_and_diversity(self) -> None:
        merger = EvidenceMerger(max_total_evidence=4, max_evidence_per_subquery=2)

        sq1 = ResearchSubQuery(subquery_id="sq_001", text="q1", purpose="P1", priority=1)
        sq2 = ResearchSubQuery(subquery_id="sq_002", text="q2", purpose="P2", priority=1)

        passages_sq1 = [_make_passage(f"sq1_p_{i}", f"doc_sq1_{i}", f"Text sq1 {i}", 0.9 - i * 0.05) for i in range(5)]
        passages_sq2 = [_make_passage(f"sq2_p_{i}", f"doc_sq2_{i}", f"Text sq2 {i}", 0.85 - i * 0.05) for i in range(5)]

        resp1 = _make_response("q1", passages_sq1)
        resp2 = _make_response("q2", passages_sq2)

        plan = ResearchPlan(
            original_query="original",
            analysis=QueryAnalysis(original_query="original"),
            subqueries=[sq1, sq2],
        )

        merged = merger.merge(plan, [(sq1, resp1), (sq2, resp2)])

        assert len(merged.selected_passages) == 4
        # Ranks must be 1 to 4
        assert [p.rank for p in merged.selected_passages] == [1, 2, 3, 4]

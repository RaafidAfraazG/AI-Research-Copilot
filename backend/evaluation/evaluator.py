"""
Evaluator for measuring and comparing retrieval and reranking quality.
Compares Hybrid Retrieval vs. Hybrid + Cross-Encoder Reranking.
"""
from __future__ import annotations

import time
from typing import Any
from pydantic import BaseModel, Field
import structlog

from evaluation.dataset import RESEARCH_EVALUATION_DATASET, EvaluationQuery
from evaluation.metrics import mrr, ndcg_at_k, recall_at_k
from ranking.ranking_service import RankingService
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievalQuery

logger = structlog.get_logger(__name__)


class MetricScore(BaseModel):
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0


class QueryEvaluationResult(BaseModel):
    query_id: str
    query_text: str
    hybrid_metrics: MetricScore
    reranked_metrics: MetricScore
    hybrid_top_sources: list[str]
    reranked_top_sources: list[str]


class EvaluationReport(BaseModel):
    total_queries: int
    hybrid_mean_metrics: MetricScore
    reranked_mean_metrics: MetricScore
    query_results: list[QueryEvaluationResult]
    elapsed_seconds: float


class RetrievalEvaluator:
    """
    Executes benchmark evaluation against the curated research dataset.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        ranking_service: RankingService | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.ranking_service = ranking_service or RankingService()

    async def evaluate(
        self,
        dataset: list[EvaluationQuery] | None = None,
        candidate_k: int = 25,
    ) -> EvaluationReport:
        """
        Evaluate all dataset queries and compare Hybrid vs. Hybrid + Reranking.
        """
        start_time = time.perf_counter()
        queries = dataset or RESEARCH_EVALUATION_DATASET
        results: list[QueryEvaluationResult] = []

        for item in queries:
            rel_set = set(item.relevant_source_ids)

            # 1. Run Hybrid Retrieval
            retrieval_query = RetrievalQuery(query_text=item.query_text, top_k=candidate_k)
            hybrid_res = await self.hybrid_retriever.search(retrieval_query)
            hybrid_chunks = hybrid_res.chunks

            # Extract source IDs from hybrid candidates
            hybrid_sources = [c.source_id for c in hybrid_chunks]

            # Compute hybrid metrics
            h_metrics = MetricScore(
                recall_at_5=round(recall_at_k(hybrid_sources, rel_set, k=5), 4),
                recall_at_10=round(recall_at_k(hybrid_sources, rel_set, k=10), 4),
                mrr=round(mrr(hybrid_sources, rel_set), 4),
                ndcg_at_10=round(ndcg_at_k(hybrid_sources, rel_set, k=10), 4),
            )

            # 2. Run Cross-Encoder Reranking
            reranked_chunks, _ = await self.ranking_service.rerank_candidates(
                query=item.query_text,
                candidates=hybrid_chunks,
                top_k=candidate_k,
            )
            reranked_sources = [c.source_id for c in reranked_chunks]

            # Compute reranked metrics
            r_metrics = MetricScore(
                recall_at_5=round(recall_at_k(reranked_sources, rel_set, k=5), 4),
                recall_at_10=round(recall_at_k(reranked_sources, rel_set, k=10), 4),
                mrr=round(mrr(reranked_sources, rel_set), 4),
                ndcg_at_10=round(ndcg_at_k(reranked_sources, rel_set, k=10), 4),
            )

            results.append(
                QueryEvaluationResult(
                    query_id=item.query_id,
                    query_text=item.query_text,
                    hybrid_metrics=h_metrics,
                    reranked_metrics=r_metrics,
                    hybrid_top_sources=hybrid_sources[:5],
                    reranked_top_sources=reranked_sources[:5],
                )
            )

        n = len(results) if results else 1
        hybrid_mean = MetricScore(
            recall_at_5=round(sum(r.hybrid_metrics.recall_at_5 for r in results) / n, 4),
            recall_at_10=round(sum(r.hybrid_metrics.recall_at_10 for r in results) / n, 4),
            mrr=round(sum(r.hybrid_metrics.mrr for r in results) / n, 4),
            ndcg_at_10=round(sum(r.hybrid_metrics.ndcg_at_10 for r in results) / n, 4),
        )
        reranked_mean = MetricScore(
            recall_at_5=round(sum(r.reranked_metrics.recall_at_5 for r in results) / n, 4),
            recall_at_10=round(sum(r.reranked_metrics.recall_at_10 for r in results) / n, 4),
            mrr=round(sum(r.reranked_metrics.mrr for r in results) / n, 4),
            ndcg_at_10=round(sum(r.reranked_metrics.ndcg_at_10 for r in results) / n, 4),
        )

        elapsed = round(time.perf_counter() - start_time, 2)

        return EvaluationReport(
            total_queries=len(results),
            hybrid_mean_metrics=hybrid_mean,
            reranked_mean_metrics=reranked_mean,
            query_results=results,
            elapsed_seconds=elapsed,
        )

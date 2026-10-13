"""
System Evaluation Engine for AI Research Copilot (Phase 7).
Orchestrates systematic, multi-variant evaluation across retrieval, ranking,
query decomposition, grounded synthesis, citation verification, and latency.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Any, Literal
from pydantic import BaseModel, Field
import structlog

from core.config import Settings, get_settings
from evaluation.answer_metrics import AnswerEvaluationScore, AnswerMetricsEvaluator
from evaluation.dataset import EvaluationQuery
from evaluation.metrics import (
    DistributionStats,
    compute_distribution_stats,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from llm.base import LLMProvider
from llm.factory import get_llm_provider
from query.models import MergedEvidencePack
from query.planner import (
    ResearchPlanRetrievalRequest,
    ResearchPlanningService,
)
from ranking.cross_encoder_reranker import CrossEncoderReranker
from ranking.ranking_service import RankingService
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievalQuery, ScoredChunk
from services.research_retrieval_service import (
    ResearchRetrievalRequest,
    ResearchRetrievalService,
)
from synthesis.answer_service import (
    ResearchAnswerRequest,
    ResearchAnswerService,
)
from synthesis.models import ResearchAnswer

logger = structlog.get_logger(__name__)


class BenchmarkQueryItem(BaseModel):
    """Unified item representation for retrieval and answer benchmarking."""
    query_id: str
    query: str
    intent: str = "FACTUAL"
    difficulty: str = "simple"
    requires_decomposition: bool = False
    relevant_documents: list[str] = Field(default_factory=list)
    relevant_passages: list[str] = Field(default_factory=list)
    required_dimensions: list[str] = Field(default_factory=list)
    expected_dimensions: list[str] = Field(default_factory=list)
    ground_truth_source: str = ""


class QueryMetricScore(BaseModel):
    """Retrieval metrics for a single query execution."""
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    evidence_dimension_coverage: float = 0.0
    retrieved_count: int = 0
    top_source_ids: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class SingleQueryEvaluation(BaseModel):
    """Comprehensive evaluation record for one query under one system variant."""
    query_id: str
    query_text: str
    intent: str
    difficulty: str
    retrieval_metrics: QueryMetricScore
    answer_metrics: AnswerEvaluationScore | None = None
    latency_breakdown: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class VariantEvaluationResult(BaseModel):
    """Aggregated evaluation results for a specific system variant."""
    variant_id: str
    variant_name: str
    description: str
    total_queries: int
    executed_queries: int
    reranker_mode: str = "none"
    is_neural_reranker: bool = False
    llm_provider: str = "none"
    is_real_llm: bool = False
    metric_distributions: dict[str, DistributionStats] = Field(default_factory=dict)
    mean_retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    mean_answer_metrics: dict[str, float] = Field(default_factory=dict)
    mean_latency_metrics: dict[str, float] = Field(default_factory=dict)
    query_evaluations: list[SingleQueryEvaluation] = Field(default_factory=list)
    total_wall_clock_ms: float = 0.0


class SystemBenchmarkReport(BaseModel):
    """Complete, machine-readable multi-variant benchmark report."""
    benchmark_name: str = "ai_research_copilot_evaluation_phase_7"
    version: str = "1.0"
    timestamp: str
    environment: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    variants: dict[str, VariantEvaluationResult] = Field(default_factory=dict)
    ablation_summary: list[dict[str, Any]] = Field(default_factory=list)


def _normalize_doc_id(raw: str) -> str:
    """Normalize document/source IDs by stripping prefixes and lowercasing."""
    cleaned = raw.strip().lower()
    if cleaned.startswith("arxiv:"):
        cleaned = cleaned[6:]
    return cleaned.strip()


def _compute_dimension_coverage(passages: list[str], dimensions: list[str]) -> float:
    """
    Calculate fraction of required/expected dimensions appearing in retrieved text passages.
    """
    if not dimensions:
        return 1.0
    if not passages:
        return 0.0

    joined_text = " ".join(passages).lower()
    matched = 0
    for dim in dimensions:
        dim_tokens = [t for t in dim.lower().split() if len(t) > 2]
        if not dim_tokens:
            continue
        if all(token in joined_text for token in dim_tokens):
            matched += 1
        elif any(token in joined_text for token in dim_tokens) and len(dim_tokens) > 1:
            # Partial credit if key keywords are present
            matched += 0.5

    return round(min(1.0, matched / len(dimensions)), 4)


class SystemEvaluator:
    """
    Evaluator executing structured multi-variant benchmark runs over research datasets.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        bm25_retriever: BM25Retriever | None = None,
        dense_retriever: DenseRetriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        ranking_service: RankingService | None = None,
        retrieval_service: ResearchRetrievalService | None = None,
        planning_service: ResearchPlanningService | None = None,
        answer_service: ResearchAnswerService | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bm25_retriever = bm25_retriever or BM25Retriever(settings=self.settings)
        self.dense_retriever = dense_retriever or DenseRetriever(settings=self.settings)
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            dense_retriever=self.dense_retriever,
            settings=self.settings,
        )
        self.ranking_service = ranking_service or RankingService()
        self.retrieval_service = retrieval_service or ResearchRetrievalService(
            hybrid_retriever=self.hybrid_retriever,
            ranking_service=self.ranking_service,
            settings=self.settings,
        )
        if llm_provider is not None:
            self.llm_provider = llm_provider
        else:
            from llm.mock_provider import MockLLMProvider
            self.llm_provider = MockLLMProvider()

        self.planning_service = planning_service or ResearchPlanningService(
            retrieval_service=self.retrieval_service,
            llm_provider=self.llm_provider,
            settings=self.settings,
        )
        self.answer_service = answer_service or ResearchAnswerService(
            retrieval_service=self.retrieval_service,
            planning_service=self.planning_service,
            llm_provider=self.llm_provider,
            settings=self.settings,
        )

    def load_dataset(self, file_path: str | Path | None = None) -> list[BenchmarkQueryItem]:
        """
        Load benchmark queries from JSON file or fallback to baseline dataset.
        """
        path = Path(file_path) if file_path else Path("data/evaluation/retrieval_benchmark.json")
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / path
        self.dataset_path: Path | None = path

        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                queries_raw = data.get("queries", [])
                items = []
                for q in queries_raw:
                    items.append(
                        BenchmarkQueryItem(
                            query_id=q.get("query_id", ""),
                            query=q.get("query", ""),
                            intent=q.get("intent", "FACTUAL"),
                            difficulty=q.get("difficulty", "simple"),
                            requires_decomposition=q.get("requires_decomposition", False),
                            relevant_documents=q.get("relevant_documents", []),
                            relevant_passages=q.get("relevant_passages", []),
                            required_dimensions=q.get("required_dimensions", []),
                            expected_dimensions=q.get("expected_dimensions", []),
                            ground_truth_source=q.get("ground_truth_source", ""),
                        )
                    )
                logger.info("benchmark_dataset_loaded", count=len(items), path=str(path))
                return items
            except Exception as e:
                logger.warning("benchmark_dataset_load_failed_falling_back", error=str(e), path=str(path))

        # Fallback to in-memory items
        from evaluation.dataset import RESEARCH_EVALUATION_DATASET
        items = []
        for eq in RESEARCH_EVALUATION_DATASET:
            items.append(
                BenchmarkQueryItem(
                    query_id=eq.query_id,
                    query=eq.query_text,
                    intent="FACTUAL",
                    difficulty="simple",
                    relevant_documents=eq.relevant_source_ids,
                    ground_truth_source=eq.description,
                )
            )
        return items

    def inspect_reranker_status(self) -> tuple[str, bool]:
        """
        Check whether the real neural cross-encoder model is actually loaded or fallback is active.
        """
        reranker = getattr(self.ranking_service, "reranker", None)
        if reranker is None:
            return "none", False

        if isinstance(reranker, CrossEncoderReranker):
            # Check if model object is present or can load
            if reranker._model is not None:
                return f"neural_loaded ({reranker.model_name})", True
            if reranker._load_failed:
                return f"fallback_active ({reranker.fallback_mode})", False
            
            # Attempt lazy load
            model = reranker._load_model()
            if model is not None:
                return f"neural_loaded ({reranker.model_name})", True
            return f"fallback_active ({reranker.fallback_mode})", False

        return "custom_reranker", False

    def inspect_llm_status(self) -> tuple[str, bool]:
        """
        Check whether real LLM inference is active or deterministic mock is used.
        """
        provider_name = self.llm_provider.provider_name if hasattr(self.llm_provider, "provider_name") else "unknown"
        is_mock = "mock" in provider_name.lower()
        return provider_name, not is_mock

    def _score_retrieved_chunks(
        self,
        chunks: list[ScoredChunk],
        item: BenchmarkQueryItem,
        latency_ms: float = 0.0,
    ) -> QueryMetricScore:
        """
        Compute standard IR retrieval metrics for a retrieved chunk list against ground truth.
        """
        rel_set = {_normalize_doc_id(d) for d in item.relevant_documents if _normalize_doc_id(d)}
        retrieved_ids = [_normalize_doc_id(c.source_id) for c in chunks]
        
        # Dimensions for coverage
        dimensions = item.required_dimensions or item.expected_dimensions
        passages = [c.content for c in chunks]
        dim_coverage = _compute_dimension_coverage(passages, dimensions)

        return QueryMetricScore(
            recall_at_1=round(recall_at_k(retrieved_ids, rel_set, k=1), 4),
            recall_at_5=round(recall_at_k(retrieved_ids, rel_set, k=5), 4),
            recall_at_10=round(recall_at_k(retrieved_ids, rel_set, k=10), 4),
            precision_at_1=round(precision_at_k(retrieved_ids, rel_set, k=1), 4),
            precision_at_5=round(precision_at_k(retrieved_ids, rel_set, k=5), 4),
            precision_at_10=round(precision_at_k(retrieved_ids, rel_set, k=10), 4),
            mrr=round(mrr(retrieved_ids, rel_set), 4),
            ndcg_at_5=round(ndcg_at_k(retrieved_ids, rel_set, k=5), 4),
            ndcg_at_10=round(ndcg_at_k(retrieved_ids, rel_set, k=10), 4),
            evidence_dimension_coverage=dim_coverage,
            retrieved_count=len(chunks),
            top_source_ids=retrieved_ids[:5],
            latency_ms=round(latency_ms, 2),
        )

    async def evaluate_variant_a_bm25(
        self,
        dataset: list[BenchmarkQueryItem],
        candidate_k: int = 25,
    ) -> VariantEvaluationResult:
        """Variant A: Pure BM25 Lexical Keyword Retrieval."""
        start_all = time.perf_counter()
        query_evals: list[SingleQueryEvaluation] = []

        for item in dataset:
            t0 = time.perf_counter()
            retrieval_query = RetrievalQuery(query_text=item.query, top_k=candidate_k)
            result = await self.bm25_retriever.search(retrieval_query)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            metrics = self._score_retrieved_chunks(result.chunks, item, latency_ms=elapsed_ms)
            query_evals.append(
                SingleQueryEvaluation(
                    query_id=item.query_id,
                    query_text=item.query,
                    intent=item.intent,
                    difficulty=item.difficulty,
                    retrieval_metrics=metrics,
                    latency_breakdown={"retrieval_latency_ms": elapsed_ms, "total_latency_ms": elapsed_ms},
                )
            )

        wall_clock_ms = (time.perf_counter() - start_all) * 1000
        return self._aggregate_variant_results(
            variant_id="variant_a_bm25",
            variant_name="Variant A: BM25 Lexical Retrieval",
            description="Pure keyword search via OpenSearch / in-memory inverted index",
            total_queries=len(dataset),
            query_evals=query_evals,
            wall_clock_ms=wall_clock_ms,
            reranker_mode="none",
            is_neural_reranker=False,
        )

    async def evaluate_variant_b_dense(
        self,
        dataset: list[BenchmarkQueryItem],
        candidate_k: int = 25,
    ) -> VariantEvaluationResult:
        """Variant B: Pure Dense Semantic Embedding Search."""
        start_all = time.perf_counter()
        query_evals: list[SingleQueryEvaluation] = []

        for item in dataset:
            t0 = time.perf_counter()
            retrieval_query = RetrievalQuery(query_text=item.query, top_k=candidate_k)
            result = await self.dense_retriever.search(retrieval_query)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            metrics = self._score_retrieved_chunks(result.chunks, item, latency_ms=elapsed_ms)
            query_evals.append(
                SingleQueryEvaluation(
                    query_id=item.query_id,
                    query_text=item.query,
                    intent=item.intent,
                    difficulty=item.difficulty,
                    retrieval_metrics=metrics,
                    latency_breakdown={"retrieval_latency_ms": elapsed_ms, "total_latency_ms": elapsed_ms},
                )
            )

        wall_clock_ms = (time.perf_counter() - start_all) * 1000
        return self._aggregate_variant_results(
            variant_id="variant_b_dense",
            variant_name="Variant B: Dense Vector Retrieval",
            description="Pure dense semantic embedding search via Qdrant / in-memory vector index",
            total_queries=len(dataset),
            query_evals=query_evals,
            wall_clock_ms=wall_clock_ms,
            reranker_mode="none",
            is_neural_reranker=False,
        )

    async def evaluate_variant_c_hybrid(
        self,
        dataset: list[BenchmarkQueryItem],
        candidate_k: int = 25,
        alpha: float = 0.5,
    ) -> VariantEvaluationResult:
        """Variant C: Hybrid Retrieval (BM25 + Dense Fusion)."""
        start_all = time.perf_counter()
        query_evals: list[SingleQueryEvaluation] = []

        for item in dataset:
            t0 = time.perf_counter()
            retrieval_query = RetrievalQuery(query_text=item.query, top_k=candidate_k, alpha=alpha)
            result = await self.hybrid_retriever.search(retrieval_query)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            metrics = self._score_retrieved_chunks(result.chunks, item, latency_ms=elapsed_ms)
            query_evals.append(
                SingleQueryEvaluation(
                    query_id=item.query_id,
                    query_text=item.query,
                    intent=item.intent,
                    difficulty=item.difficulty,
                    retrieval_metrics=metrics,
                    latency_breakdown={"retrieval_latency_ms": elapsed_ms, "total_latency_ms": elapsed_ms},
                )
            )

        wall_clock_ms = (time.perf_counter() - start_all) * 1000
        return self._aggregate_variant_results(
            variant_id="variant_c_hybrid",
            variant_name="Variant C: Hybrid Retrieval (BM25 + Dense)",
            description="Lexical and dense fusion using Reciprocal Rank Fusion (RRF)",
            total_queries=len(dataset),
            query_evals=query_evals,
            wall_clock_ms=wall_clock_ms,
            reranker_mode="none",
            is_neural_reranker=False,
        )

    async def evaluate_variant_d_hybrid_rerank(
        self,
        dataset: list[BenchmarkQueryItem],
        candidate_k: int = 25,
        alpha: float = 0.5,
    ) -> VariantEvaluationResult:
        """Variant D: Hybrid Retrieval + Cross-Encoder Reranking."""
        start_all = time.perf_counter()
        query_evals: list[SingleQueryEvaluation] = []
        reranker_mode, is_neural = self.inspect_reranker_status()

        for item in dataset:
            t0 = time.perf_counter()
            retrieval_query = RetrievalQuery(query_text=item.query, top_k=candidate_k, alpha=alpha)
            hybrid_res = await self.hybrid_retriever.search(retrieval_query)
            hybrid_ms = (time.perf_counter() - t0) * 1000

            t_rerank = time.perf_counter()
            reranked_chunks, rerank_ms = await self.ranking_service.rerank_candidates(
                query=item.query,
                candidates=hybrid_res.chunks,
                top_k=candidate_k,
            )
            total_ms = (time.perf_counter() - t0) * 1000

            metrics = self._score_retrieved_chunks(reranked_chunks, item, latency_ms=total_ms)
            notes = []
            if not is_neural:
                notes.append("Cross-encoder benchmark unavailable in current environment. Fallback ranking was used.")

            query_evals.append(
                SingleQueryEvaluation(
                    query_id=item.query_id,
                    query_text=item.query,
                    intent=item.intent,
                    difficulty=item.difficulty,
                    retrieval_metrics=metrics,
                    latency_breakdown={
                        "retrieval_latency_ms": hybrid_ms,
                        "reranking_latency_ms": rerank_ms,
                        "total_latency_ms": total_ms,
                    },
                    notes=notes,
                )
            )

        wall_clock_ms = (time.perf_counter() - start_all) * 1000
        return self._aggregate_variant_results(
            variant_id="variant_d_hybrid_reranker",
            variant_name="Variant D: Hybrid + Cross-Encoder Reranker",
            description="Two-stage: Hybrid candidate search followed by Cross-Encoder scoring",
            total_queries=len(dataset),
            query_evals=query_evals,
            wall_clock_ms=wall_clock_ms,
            reranker_mode=reranker_mode,
            is_neural_reranker=is_neural,
        )

    async def evaluate_variant_e_decomp_hybrid(
        self,
        dataset: list[BenchmarkQueryItem],
        candidate_k: int = 25,
        evidence_k: int = 6,
    ) -> VariantEvaluationResult:
        """Variant E: Query Decomposition + Parallel Hybrid Retrieval + Evidence Merging."""
        start_all = time.perf_counter()
        query_evals: list[SingleQueryEvaluation] = []

        for item in dataset:
            t0 = time.perf_counter()
            req = ResearchPlanRetrievalRequest(
                query=item.query,
                enable_decomposition=True,
                decomposition_mode="auto",
                candidate_k=candidate_k,
                evidence_k=evidence_k,
            )
            merged_pack, plan_lat, plan = await self.planning_service.plan_and_retrieve(req)
            total_ms = (time.perf_counter() - t0) * 1000

            # Convert merged evidence passages to scored chunks for IR metric calculation
            all_chunks: list[ScoredChunk] = []
            for item_ev in merged_pack.selected_passages:
                all_chunks.append(
                    ScoredChunk(
                        chunk_id=item_ev.passage_id,
                        document_id=item_ev.document_id,
                        source_type=item_ev.source_type,
                        source_id=item_ev.source_id,
                        title=item_ev.title,
                        url=item_ev.url,
                        content=item_ev.text,
                        score=item_ev.relevance_score,
                    )
                )

            metrics = self._score_retrieved_chunks(all_chunks, item, latency_ms=total_ms)
            query_evals.append(
                SingleQueryEvaluation(
                    query_id=item.query_id,
                    query_text=item.query,
                    intent=item.intent,
                    difficulty=item.difficulty,
                    retrieval_metrics=metrics,
                    latency_breakdown={
                        "query_analysis_latency_ms": plan_lat.query_analysis_latency_ms,
                        "decomposition_latency_ms": plan_lat.decomposition_latency_ms,
                        "subquery_retrieval_wall_clock_ms": plan_lat.subquery_retrieval_wall_clock_ms,
                        "sum_of_subquery_latencies_ms": plan_lat.sum_of_subquery_latencies_ms,
                        "evidence_merge_latency_ms": plan_lat.evidence_merge_latency_ms,
                        "total_latency_ms": total_ms,
                    },
                    notes=[f"Subqueries: {len(plan.subqueries)}", f"Strategy: {plan.strategy}"],
                )
            )

        wall_clock_ms = (time.perf_counter() - start_all) * 1000
        return self._aggregate_variant_results(
            variant_id="variant_e_decomp_hybrid",
            variant_name="Variant E: Query Decomposition + Hybrid",
            description="Intent-driven subquery decomposition, parallel hybrid retrieval, and evidence merging",
            total_queries=len(dataset),
            query_evals=query_evals,
            wall_clock_ms=wall_clock_ms,
            reranker_mode="none",
            is_neural_reranker=False,
        )

    async def evaluate_variant_f_full_pipeline(
        self,
        dataset: list[BenchmarkQueryItem],
        candidate_k: int = 25,
        evidence_k: int = 6,
    ) -> VariantEvaluationResult:
        """Variant F: Full Research Pipeline (Planning -> Retrieval -> Rerank -> Grounded Synthesis -> Verification)."""
        start_all = time.perf_counter()
        query_evals: list[SingleQueryEvaluation] = []
        reranker_mode, is_neural = self.inspect_reranker_status()
        llm_name, is_real_llm = self.inspect_llm_status()

        for item in dataset:
            t0 = time.perf_counter()
            req = ResearchAnswerRequest(
                query=item.query,
                enable_decomposition=True,
                decomposition_mode="auto",
                candidate_k=candidate_k,
                evidence_k=evidence_k,
                provider_override=self.llm_provider.provider_name,
            )
            answer: ResearchAnswer = await self.answer_service.answer_question(req)
            total_ms = (time.perf_counter() - t0) * 1000

            # Retrieval metric scoring from cited sources / citations
            cited_chunks: list[ScoredChunk] = []
            for cite in answer.citations:
                cited_chunks.append(
                    ScoredChunk(
                        chunk_id=cite.citation_id,
                        document_id=cite.source_id,
                        source_type=cite.source_type,
                        source_id=cite.source_id,
                        title=cite.title,
                        url=cite.url,
                        content=cite.formatted_citation,
                        score=1.0,
                    )
                )
            if not cited_chunks:
                for src in answer.sources:
                    cited_chunks.append(
                        ScoredChunk(
                            chunk_id=src.source_id,
                            document_id=src.source_id,
                            source_type=src.source_type,
                            source_id=src.source_id,
                            title=src.title,
                            url=src.url,
                            content=src.title,
                            score=1.0,
                        )
                    )

            ret_metrics = self._score_retrieved_chunks(cited_chunks, item, latency_ms=total_ms)

            # Answer metric scoring
            expected_dims = item.expected_dimensions or item.required_dimensions
            ans_score = AnswerMetricsEvaluator.evaluate_answer(
                answer=answer,
                query_id=item.query_id,
                expected_dimensions=expected_dims,
                relevant_documents=item.relevant_documents,
            )

            notes = []
            if not is_neural:
                notes.append("Reranker running in fallback mode.")
            if not is_real_llm:
                notes.append("Evaluated with deterministic Mock LLM provider.")

            query_evals.append(
                SingleQueryEvaluation(
                    query_id=item.query_id,
                    query_text=item.query,
                    intent=item.intent,
                    difficulty=item.difficulty,
                    retrieval_metrics=ret_metrics,
                    answer_metrics=ans_score,
                    latency_breakdown={
                        "planning_latency_ms": answer.metadata.planning_latency_ms,
                        "retrieval_latency_ms": answer.metadata.retrieval_latency_ms,
                        "generation_latency_ms": answer.metadata.generation_latency_ms,
                        "verification_latency_ms": answer.metadata.verification_latency_ms,
                        "total_latency_ms": answer.metadata.total_latency_ms or total_ms,
                    } if answer.metadata else {"total_latency_ms": total_ms},
                    notes=notes,
                )
            )

        wall_clock_ms = (time.perf_counter() - start_all) * 1000
        return self._aggregate_variant_results(
            variant_id="variant_f_full_pipeline",
            variant_name="Variant F: Full Research Pipeline",
            description="End-to-End: Query Intelligence -> Parallel Hybrid -> Rerank -> Grounded Synthesis -> Verification",
            total_queries=len(dataset),
            query_evals=query_evals,
            wall_clock_ms=wall_clock_ms,
            reranker_mode=reranker_mode,
            is_neural_reranker=is_neural,
            llm_provider=llm_name,
            is_real_llm=is_real_llm,
        )

    def _aggregate_variant_results(
        self,
        variant_id: str,
        variant_name: str,
        description: str,
        total_queries: int,
        query_evals: list[SingleQueryEvaluation],
        wall_clock_ms: float,
        reranker_mode: str = "none",
        is_neural_reranker: bool = False,
        llm_provider: str = "none",
        is_real_llm: bool = False,
    ) -> VariantEvaluationResult:
        """Compute mean metrics, statistical distributions, and aggregate summary."""
        n = len(query_evals) if query_evals else 1

        # Retrieval distributions
        ret_keys = [
            "recall_at_1", "recall_at_5", "recall_at_10",
            "precision_at_1", "precision_at_5", "precision_at_10",
            "mrr", "ndcg_at_5", "ndcg_at_10", "evidence_dimension_coverage", "latency_ms"
        ]
        metric_dists: dict[str, DistributionStats] = {}
        mean_ret: dict[str, float] = {}

        for k in ret_keys:
            vals = [getattr(q.retrieval_metrics, k, 0.0) for q in query_evals]
            stats = compute_distribution_stats(vals)
            metric_dists[k] = stats
            mean_ret[k] = stats.mean

        # Answer distributions (if present)
        mean_ans: dict[str, float] = {}
        has_ans = any(q.answer_metrics is not None for q in query_evals)
        if has_ans:
            ans_keys = [
                "claim_support_rate", "unsupported_claim_rate", "citation_coverage",
                "citation_validity_rate", "citation_precision", "citation_traceability_rate",
                "evidence_dimension_coverage"
            ]
            for ak in ans_keys:
                vals = [
                    getattr(q.answer_metrics, ak, 0.0)
                    for q in query_evals
                    if q.answer_metrics is not None
                ]
                stats = compute_distribution_stats(vals)
                metric_dists[f"ans_{ak}"] = stats
                mean_ans[ak] = stats.mean

        # Latency means
        mean_lat: dict[str, float] = {}
        all_lat_keys: set[str] = set()
        for q in query_evals:
            all_lat_keys.update(q.latency_breakdown.keys())
        for lk in all_lat_keys:
            vals = [q.latency_breakdown.get(lk, 0.0) for q in query_evals if lk in q.latency_breakdown]
            if vals:
                mean_lat[lk] = round(sum(vals) / len(vals), 2)

        return VariantEvaluationResult(
            variant_id=variant_id,
            variant_name=variant_name,
            description=description,
            total_queries=total_queries,
            executed_queries=len(query_evals),
            reranker_mode=reranker_mode,
            is_neural_reranker=is_neural_reranker,
            llm_provider=llm_provider,
            is_real_llm=is_real_llm,
            metric_distributions=metric_dists,
            mean_retrieval_metrics=mean_ret,
            mean_answer_metrics=mean_ans,
            mean_latency_metrics=mean_lat,
            query_evaluations=query_evals,
            total_wall_clock_ms=round(wall_clock_ms, 2),
        )

    def collect_environment(self, dataset_size: int) -> dict[str, Any]:
        """Identity of everything the numbers depend on (no filesystem paths or secrets)."""
        import hashlib

        from retrieval.index_manifest import IndexIntegrityError, IndexStore

        env: dict[str, Any] = {"dataset_size": dataset_size}
        path = getattr(self, "dataset_path", None)
        if path is not None and Path(path).exists():
            try:
                raw = Path(path).read_bytes()
                data = json.loads(raw.decode("utf-8"))
                env["dataset"] = {
                    "name": data.get("benchmark_name"),
                    "version": data.get("version"),
                    "file": Path(path).name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "queries": len(data.get("queries", [])),
                }
            except Exception:
                env["dataset"] = {"file": Path(path).name}
        try:
            manifest = IndexStore(self.settings).active_manifest()
            env["index"] = {
                "generation_id": manifest.generation_id,
                "created_at": manifest.created_at,
                "documents": manifest.counts.documents,
                "chunks": manifest.counts.chunks,
                "chunking": manifest.chunking.model_dump(),
                "corpus_sha256": manifest.corpus.get("sha256"),
            }
            env["embedding"] = {
                "provider": manifest.embedding.provider,
                "model": manifest.embedding.model_name,
                "dimension": manifest.embedding.dimension,
                "revision": manifest.embedding.revision,
                "is_mock": manifest.embedding.is_mock,
            }
        except IndexIntegrityError as e:
            env["index"] = {"error": e.code}
        rr = getattr(self.ranking_service, "reranker", None)
        if rr is not None:
            env["reranker_model"] = getattr(rr, "model_name", None)
            env["reranker_device"] = getattr(rr, "device", None)
        return env

    async def run_full_benchmark(
        self,
        dataset: list[BenchmarkQueryItem] | None = None,
        variants_to_run: list[str] | None = None,
        candidate_k: int = 25,
        evidence_k: int = 6,
    ) -> SystemBenchmarkReport:
        """
        Execute benchmark evaluation across all specified variants and build SystemBenchmarkReport.
        """
        items = dataset or self.load_dataset()
        target_variants = variants_to_run or ["a", "b", "c", "d", "e", "f"]
        results: dict[str, VariantEvaluationResult] = {}
        ablation_rows: list[dict[str, Any]] = []

        reranker_mode, is_neural = self.inspect_reranker_status()
        llm_mode, is_real_llm = self.inspect_llm_status()

        env_info = self.collect_environment(len(items))
        idx = env_info.get("index", {})
        emb = env_info.get("embedding", {})
        limitations = [
            f"Ground truth based on {len(items)} curated technical queries targeting a small indexed corpus.",
            "Sample size is directional and not statistically conclusive.",
            (
                f"The index contains only {idx.get('documents', '?')} documents / {idx.get('chunks', '?')} chunks; "
                "with so few candidates, Recall@5/@10 saturate and cannot separate retrievers. "
                "Rank-sensitive metrics (Recall@1, MRR, NDCG) are the informative ones."
            ),
            "Relevance labels are document-level and author-assigned; no human adjudication or inter-annotator agreement.",
        ]
        if emb.get("is_mock"):
            limitations.append("DENSE RESULTS USE MOCK (HASH) EMBEDDINGS and are not a measurement of a real model.")
        if not is_neural:
            limitations.append(
                "Cross-encoder benchmark unavailable in current environment. Fallback ranking was used."
            )
        if not is_real_llm:
            limitations.append(
                "Answer evaluation executed with deterministic Mock LLM provider. Live model variance not measured."
            )

        # 1. Variant A (BM25)
        if "a" in target_variants or "bm25" in target_variants:
            logger.info("evaluating_variant_a_bm25")
            res_a = await self.evaluate_variant_a_bm25(items, candidate_k=candidate_k)
            results["variant_a_bm25"] = res_a
            ablation_rows.append(self._make_ablation_row(res_a))

        # 2. Variant B (Dense)
        if "b" in target_variants or "dense" in target_variants:
            logger.info("evaluating_variant_b_dense")
            res_b = await self.evaluate_variant_b_dense(items, candidate_k=candidate_k)
            results["variant_b_dense"] = res_b
            ablation_rows.append(self._make_ablation_row(res_b))

        # 3. Variant C (Hybrid)
        if "c" in target_variants or "hybrid" in target_variants:
            logger.info("evaluating_variant_c_hybrid")
            res_c = await self.evaluate_variant_c_hybrid(items, candidate_k=candidate_k)
            results["variant_c_hybrid"] = res_c
            ablation_rows.append(self._make_ablation_row(res_c))

        # 4. Variant D (Hybrid + Reranker)
        if "d" in target_variants or "hybrid_rerank" in target_variants or "rerank" in target_variants:
            logger.info("evaluating_variant_d_hybrid_reranker")
            res_d = await self.evaluate_variant_d_hybrid_rerank(items, candidate_k=candidate_k)
            results["variant_d_hybrid_reranker"] = res_d
            ablation_rows.append(self._make_ablation_row(res_d))

        # 5. Variant E (Decomposition + Hybrid)
        if "e" in target_variants or "decomp" in target_variants or "decomp_hybrid" in target_variants:
            logger.info("evaluating_variant_e_decomp_hybrid")
            res_e = await self.evaluate_variant_e_decomp_hybrid(items, candidate_k=candidate_k, evidence_k=evidence_k)
            results["variant_e_decomp_hybrid"] = res_e
            ablation_rows.append(self._make_ablation_row(res_e))

        # 6. Variant F (Full Pipeline)
        if "f" in target_variants or "full" in target_variants or "full_pipeline" in target_variants:
            logger.info("evaluating_variant_f_full_pipeline")
            res_f = await self.evaluate_variant_f_full_pipeline(items, candidate_k=candidate_k, evidence_k=evidence_k)
            results["variant_f_full_pipeline"] = res_f
            ablation_rows.append(self._make_ablation_row(res_f))

        import datetime
        report = SystemBenchmarkReport(
            benchmark_name="ai_research_copilot_system_benchmark",
            version="1.0",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            environment={
                **env_info,
                "reranker_mode": reranker_mode,
                "is_neural_reranker": is_neural,
                "llm_provider": llm_mode,
                "is_real_llm": is_real_llm,
                "dataset_size": len(items),
            },
            limitations=limitations,
            variants=results,
            ablation_summary=ablation_rows,
        )
        return report

    def _make_ablation_row(self, res: VariantEvaluationResult) -> dict[str, Any]:
        """Format variant metrics into a clean ablation summary row."""
        return {
            "variant_id": res.variant_id,
            "variant_name": res.variant_name,
            "recall_at_1": res.mean_retrieval_metrics.get("recall_at_1", 0.0),
            "recall_at_5": res.mean_retrieval_metrics.get("recall_at_5", 0.0),
            "ndcg_at_5": res.mean_retrieval_metrics.get("ndcg_at_5", 0.0),
            "recall_at_10": res.mean_retrieval_metrics.get("recall_at_10", 0.0),
            "precision_at_5": res.mean_retrieval_metrics.get("precision_at_5", 0.0),
            "mrr": res.mean_retrieval_metrics.get("mrr", 0.0),
            "ndcg_at_10": res.mean_retrieval_metrics.get("ndcg_at_10", 0.0),
            "dimension_coverage": res.mean_retrieval_metrics.get("evidence_dimension_coverage", 0.0),
            "avg_latency_ms": res.mean_retrieval_metrics.get("latency_ms", 0.0),
            "claim_support_rate": res.mean_answer_metrics.get("claim_support_rate"),
            "citation_coverage": res.mean_answer_metrics.get("citation_coverage"),
            "citation_validity": res.mean_answer_metrics.get("citation_validity_rate"),
            "reranker_mode": res.reranker_mode,
            "is_neural_reranker": res.is_neural_reranker,
        }

    def render_markdown_report(self, report: SystemBenchmarkReport) -> str:
        """
        Generate a comprehensive, scientifically honest Markdown report.
        """
        lines: list[str] = [
            f"# AI Research Copilot — Evaluation & Ablation Report",
            f"",
            f"**Generated:** `{report.timestamp}`  ",
            f"**Dataset Queries:** `{report.environment.get('dataset_size', 0)}`  ",
            f"**Reranker Status:** `{report.environment.get('reranker_mode', 'unknown')}` (Neural: `{report.environment.get('is_neural_reranker')}`)  ",
            f"**LLM Status:** `{report.environment.get('llm_provider', 'unknown')}` (Live Model: `{report.environment.get('is_real_llm')}`)  ",
            f"**Embedding:** `{(report.environment.get('embedding') or {}).get('model')}` "
            f"({(report.environment.get('embedding') or {}).get('dimension')}d, mock: `{(report.environment.get('embedding') or {}).get('is_mock')}`, "
            f"rev `{str((report.environment.get('embedding') or {}).get('revision'))[:12]}`)  ",
            f"**Index generation:** `{(report.environment.get('index') or {}).get('generation_id')}` "
            f"({(report.environment.get('index') or {}).get('documents')} docs / {(report.environment.get('index') or {}).get('chunks')} chunks)  ",
            f"**Reranker model:** `{report.environment.get('reranker_model')}` on `{report.environment.get('reranker_device')}`  ",
            f"**Dataset:** `{(report.environment.get('dataset') or {}).get('file')}` "
            f"sha256 `{str((report.environment.get('dataset') or {}).get('sha256'))[:12]}` "
            f"v{(report.environment.get('dataset') or {}).get('version')}  ",
            f"",
            f"---",
            f"",
            f"## 1. System Retrieval Ablation Study",
            f"",
            f"| Variant | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR | NDCG@5 | NDCG@10 | Dim Coverage | Avg Latency (ms) |",
            f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for row in report.ablation_summary:
            v_name = row["variant_name"]
            r1 = f"{row['recall_at_1']:.4f}"
            n5 = f"{row['ndcg_at_5']:.4f}"
            r5 = f"{row['recall_at_5']:.4f}"
            r10 = f"{row['recall_at_10']:.4f}"
            p5 = f"{row['precision_at_5']:.4f}"
            mrr_val = f"{row['mrr']:.4f}"
            ndcg10 = f"{row['ndcg_at_10']:.4f}"
            dim_cov = f"{row['dimension_coverage']:.4f}"
            lat = f"{row['avg_latency_ms']:.2f}"
            lines.append(f"| **{v_name}** | {r1} | {r5} | {r10} | {p5} | {mrr_val} | {n5} | {ndcg10} | {dim_cov} | {lat} |")

        lines.extend([
            f"",
            f"> [!NOTE]",
            f"> Directional evaluation on a very small corpus; see section 5. Do not generalize.",
            f"",
            f"---",
            f"",
            f"## 2. Statistical Distribution of Retrieval Metrics",
            f"",
            f"| Variant | Metric | Mean | Median | Min | Max | StdDev |",
            f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ])

        for var_id, vres in report.variants.items():
            for m_key in ["recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_5", "ndcg_at_10", "evidence_dimension_coverage"]:
                d = vres.metric_distributions.get(m_key)
                if d:
                    lines.append(
                        f"| {vres.variant_name} | `{m_key}` | {d.mean:.4f} | {d.median:.4f} | {d.min_val:.4f} | {d.max_val:.4f} | {d.std_dev:.4f} |"
                    )

        # Answer Quality Section if Variant F present
        if "variant_f_full_pipeline" in report.variants:
            vf = report.variants["variant_f_full_pipeline"]
            lines.extend([
                f"",
                f"---",
                f"",
                f"## 3. Answer Quality & Citation Grounding (Full Pipeline)",
                f"",
                f"| Metric | Mean Score | Description |",
                f"| :--- | :---: | :--- |",
                f"| **Claim Support Rate** | `{vf.mean_answer_metrics.get('claim_support_rate', 0.0):.4f}` | Fraction of factual claims verified as fully supported by evidence |",
                f"| **Unsupported Claim Rate** | `{vf.mean_answer_metrics.get('unsupported_claim_rate', 0.0):.4f}` | Fraction of claims lacking sufficient citation evidence |",
                f"| **Citation Coverage** | `{vf.mean_answer_metrics.get('citation_coverage', 0.0):.4f}` | Fraction of claims possessing at least one citation |",
                f"| **Citation Validity Rate** | `{vf.mean_answer_metrics.get('citation_validity_rate', 0.0):.4f}` | Fraction of citations linking to valid retrieved passage IDs |",
                f"| **Citation Precision** | `{vf.mean_answer_metrics.get('citation_precision', 0.0):.4f}` | Fraction of citations with authentic document metadata |",
                f"| **Citation Traceability Rate** | `{vf.mean_answer_metrics.get('citation_traceability_rate', 0.0):.4f}` | Fraction of claims traceable to canonical source URLs |",
                f"| **Evidence Dimension Coverage** | `{vf.mean_answer_metrics.get('evidence_dimension_coverage', 0.0):.4f}` | Fraction of expected research dimensions found in synthesis |",
            ])

        # Latency breakdown
        lines.extend([
            f"",
            f"---",
            f"",
            f"## 4. Latency Profiling",
            f"",
            f"| Variant | Mean Total Latency (ms) | Granular Components |",
            f"| :--- | :---: | :--- |",
        ])

        for var_id, vres in report.variants.items():
            tot_lat = vres.mean_retrieval_metrics.get("latency_ms", 0.0)
            comp_str = ", ".join(f"`{k}`: {v:.1f}ms" for k, v in vres.mean_latency_metrics.items() if k != "total_latency_ms")
            lines.append(f"| **{vres.variant_name}** | {tot_lat:.2f} ms | {comp_str or 'N/A'} |")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 5. Methodological Limitations & Integrity Declarations",
            f"",
        ])
        for lim in report.limitations:
            lines.append(f"* {lim}")

        return "\n".join(lines) + "\n"

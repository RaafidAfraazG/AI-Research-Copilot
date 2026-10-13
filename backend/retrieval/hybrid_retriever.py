"""
Hybrid Retriever combining BM25 (OpenSearch) and Dense (Qdrant) search using
Reciprocal Rank Fusion (RRF) and Relative Score Fusion (RSF).
"""
from __future__ import annotations

import asyncio
import time
from typing import Literal
import structlog

from core.config import Settings, get_settings
from retrieval.base import BaseRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.models import RetrievalQuery, RetrievalResult, ScoredChunk

logger = structlog.get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Executes lexical and dense vector search in parallel, then fuses
    the candidate sets into a unified, high-relevance ranked result list.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        dense_retriever: DenseRetriever | None = None,
        rrf_k: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bm25_retriever = bm25_retriever or BM25Retriever(settings=self.settings)
        self.dense_retriever = dense_retriever or DenseRetriever(settings=self.settings)
        self.rrf_k = rrf_k or self.settings.rrf_k

    @property
    def name(self) -> str:
        return "hybrid"

    async def aclose(self) -> None:
        await self.bm25_retriever.aclose()
        await self.dense_retriever.aclose()

    async def ping(self) -> bool:
        """Check if both underlying search engines are alive."""
        bm25_ok, dense_ok = await asyncio.gather(
            self.bm25_retriever.ping(),
            self.dense_retriever.ping(),
            return_exceptions=True,
        )
        return bool(bm25_ok is True and dense_ok is True)

    def reciprocal_rank_fusion(
        self,
        bm25_chunks: list[ScoredChunk],
        dense_chunks: list[ScoredChunk],
        alpha: float = 0.5,
        k: int = 60,
    ) -> list[ScoredChunk]:
        """
        Compute Reciprocal Rank Fusion scores across BM25 and dense results.

        Formula:
            RRF_score(d) = w_bm25 / (k + rank_bm25(d)) + w_dense / (k + rank_dense(d))
        where:
            w_bm25 = 2.0 * (1.0 - alpha)
            w_dense = 2.0 * alpha
        """
        w_bm25 = 2.0 * (1.0 - alpha)
        w_dense = 2.0 * alpha

        # Map by unique chunk key: chunk_id or (document_id, chunk_index)
        chunk_map: dict[str, ScoredChunk] = {}
        rrf_scores: dict[str, float] = {}

        # 1. Process BM25 rankings
        for rank, chunk in enumerate(bm25_chunks, start=1):
            key = chunk.chunk_id or f"{chunk.document_id}:{chunk.chunk_index}"
            score_contrib = w_bm25 / (k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + score_contrib

            if key not in chunk_map:
                chunk_map[key] = chunk.model_copy()
            chunk_map[key].bm25_rank = rank
            chunk_map[key].bm25_score = chunk.score

        # 2. Process Dense rankings
        for rank, chunk in enumerate(dense_chunks, start=1):
            key = chunk.chunk_id or f"{chunk.document_id}:{chunk.chunk_index}"
            score_contrib = w_dense / (k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + score_contrib

            if key not in chunk_map:
                chunk_map[key] = chunk.model_copy()
            chunk_map[key].dense_rank = rank
            chunk_map[key].dense_score = chunk.score

        # 3. Sort by fused RRF score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        final_chunks: list[ScoredChunk] = []
        for rank, key in enumerate(sorted_keys, start=1):
            c = chunk_map[key]
            c.score = round(rrf_scores[key], 6)
            c.rank = rank
            c.retriever_name = "hybrid"
            final_chunks.append(c)

        return final_chunks

    def relative_score_fusion(
        self,
        bm25_chunks: list[ScoredChunk],
        dense_chunks: list[ScoredChunk],
        alpha: float = 0.5,
    ) -> list[ScoredChunk]:
        """
        Compute Min-Max normalized score fusion.
        """
        def normalize_scores(chunks: list[ScoredChunk]) -> dict[str, float]:
            if not chunks:
                return {}
            scores = [c.score for c in chunks]
            min_s, max_s = min(scores), max(scores)
            span = max_s - min_s
            if span <= 1e-9:
                return {c.chunk_id: 1.0 for c in chunks}
            return {c.chunk_id: (c.score - min_s) / span for c in chunks}

        bm25_norm = normalize_scores(bm25_chunks)
        dense_norm = normalize_scores(dense_chunks)

        chunk_map: dict[str, ScoredChunk] = {}
        combined_scores: dict[str, float] = {}

        for rank, chunk in enumerate(bm25_chunks, start=1):
            key = chunk.chunk_id
            chunk_map[key] = chunk.model_copy()
            chunk_map[key].bm25_rank = rank
            chunk_map[key].bm25_score = chunk.score
            combined_scores[key] = (1.0 - alpha) * bm25_norm.get(key, 0.0)

        for rank, chunk in enumerate(dense_chunks, start=1):
            key = chunk.chunk_id
            if key not in chunk_map:
                chunk_map[key] = chunk.model_copy()
            chunk_map[key].dense_rank = rank
            chunk_map[key].dense_score = chunk.score
            combined_scores[key] = combined_scores.get(key, 0.0) + alpha * dense_norm.get(key, 0.0)

        sorted_keys = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        final_chunks: list[ScoredChunk] = []
        for rank, key in enumerate(sorted_keys, start=1):
            c = chunk_map[key]
            c.score = round(combined_scores[key], 6)
            c.rank = rank
            c.retriever_name = "hybrid"
            final_chunks.append(c)

        return final_chunks

    async def search(
        self,
        query: RetrievalQuery,
        fusion_method: Literal["rrf", "rsf"] = "rrf",
    ) -> RetrievalResult:
        """
        Execute concurrent BM25 and Dense vector retrieval and merge via fusion.
        """
        start_time = time.perf_counter()

        # Fetch candidate sets (e.g. top 30 from each)
        candidate_k = max(query.top_k * 3, 30)
        bm25_query = query.model_copy(update={"top_k": candidate_k})
        dense_query = query.model_copy(update={"top_k": candidate_k})

        # Run BM25 and Dense concurrently
        bm25_res, dense_res = await asyncio.gather(
            self.bm25_retriever.search(bm25_query),
            self.dense_retriever.search(dense_query),
            return_exceptions=False,
        )

        bm25_chunks = bm25_res.chunks if isinstance(bm25_res, RetrievalResult) else []
        dense_chunks = dense_res.chunks if isinstance(dense_res, RetrievalResult) else []

        # Fusion
        if fusion_method == "rsf":
            fused_chunks = self.relative_score_fusion(
                bm25_chunks=bm25_chunks,
                dense_chunks=dense_chunks,
                alpha=query.alpha,
            )
        else:
            fused_chunks = self.reciprocal_rank_fusion(
                bm25_chunks=bm25_chunks,
                dense_chunks=dense_chunks,
                alpha=query.alpha,
                k=self.rrf_k,
            )

        # Slice to requested top_k
        top_chunks = fused_chunks[: query.top_k]
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "hybrid_search_completed",
            query=query.query_text,
            bm25_candidates=len(bm25_chunks),
            dense_candidates=len(dense_chunks),
            fused_results=len(top_chunks),
            latency_ms=latency_ms,
        )

        return RetrievalResult(
            query_text=query.query_text,
            retriever="hybrid",
            chunks=top_chunks,
            total_found=len(fused_chunks),
            latency_ms=latency_ms,
            metadata={
                "fusion_method": fusion_method,
                "rrf_k": self.rrf_k,
                "alpha": query.alpha,
                "fusion_mode": fusion_method,
                "bm25_latency_ms": getattr(bm25_res, "latency_ms", 0.0),
                "dense_latency_ms": getattr(dense_res, "latency_ms", 0.0),
                "embedding_model": getattr(dense_res, "metadata", {}).get("embedding_model"),
                "index_generation": getattr(dense_res, "metadata", {}).get("index_generation"),
            },
        )

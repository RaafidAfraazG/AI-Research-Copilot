"""
Unified Research Retrieval Service orchestrating Candidate Retrieval, Cross-Encoder Reranking,
and Evidence Selection into an EvidencePack.
"""
from __future__ import annotations

import time
from typing import Any
from pydantic import BaseModel, Field
import structlog

from core.config import Settings, get_settings
from ranking.evidence import EvidencePack
from ranking.evidence_selector import EvidenceSelector
from ranking.ranking_service import RankingService
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievalQuery, ScoredChunk

logger = structlog.get_logger(__name__)


class ResearchRetrievalRequest(BaseModel):
    """
    Request parameters for research retrieval, reranking, and evidence packaging.
    """
    query: str = Field(..., min_length=1, description="Research query string")
    candidate_k: int = Field(default=25, ge=1, le=100, description="Number of candidate passages to retrieve before reranking")
    evidence_k: int = Field(default=6, ge=1, le=20, description="Target number of top evidence passages to select")
    min_score: float | None = Field(default=None, description="Minimum relevance score threshold for evidence")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Hybrid balance: 0.0 BM25 only, 1.0 Dense only, 0.5 Balanced")
    source_types: list[str] | None = Field(default=None, description="Filter by source types (e.g. ['arxiv', 'web'])")
    filters: dict[str, Any] | None = Field(default=None, description="Additional metadata filters")


class LatencyBreakdown(BaseModel):
    """Granular latency measurements across the retrieval and ranking pipeline."""
    bm25_latency_ms: float = 0.0
    dense_latency_ms: float = 0.0
    hybrid_fusion_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    evidence_selection_latency_ms: float = 0.0
    total_retrieval_latency_ms: float = 0.0


class ResearchRetrievalResponse(BaseModel):
    """
    Complete response containing candidate pool, reranked passages,
    final EvidencePack, and latency breakdown.
    """
    query: str
    total_candidates: int
    total_evidence: int
    candidates: list[ScoredChunk]
    reranked: list[ScoredChunk]
    evidence_pack: EvidencePack
    latency_breakdown: LatencyBreakdown


class ResearchRetrievalService:
    """
    High-level service coordinating the full pipeline:
    HybridRetriever -> RankingService -> EvidenceSelector -> EvidencePack
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        ranking_service: RankingService | None = None,
        evidence_selector: EvidenceSelector | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.ranking_service = ranking_service or RankingService()
        self.evidence_selector = evidence_selector or EvidenceSelector()

    async def aclose(self) -> None:
        """Release network clients held by the hybrid retriever."""
        closer = getattr(self.hybrid_retriever, "aclose", None)
        if closer is not None:
            await closer()

    async def retrieve_evidence(
        self,
        request: ResearchRetrievalRequest,
    ) -> ResearchRetrievalResponse:
        """
        Execute candidate retrieval, cross-encoder reranking, and evidence selection.
        """
        pipeline_start = time.perf_counter()

        # ── Step 1: Hybrid Candidate Retrieval ────────────────────────────────
        retrieval_query = RetrievalQuery(
            query_text=request.query,
            top_k=request.candidate_k,
            source_types=request.source_types,
            filters=request.filters,
            alpha=request.alpha,
        )

        hybrid_result = await self.hybrid_retriever.search(retrieval_query)
        candidates = hybrid_result.chunks

        bm25_lat = float(hybrid_result.metadata.get("bm25_latency_ms", 0.0))
        dense_lat = float(hybrid_result.metadata.get("dense_latency_ms", 0.0))
        fusion_lat = float(hybrid_result.metadata.get("fusion_latency_ms", 0.0))

        # ── Step 2: Cross-Encoder Reranking ───────────────────────────────────
        reranked_chunks, rerank_lat = await self.ranking_service.rerank_candidates(
            query=request.query,
            candidates=candidates,
            top_k=request.candidate_k,
        )

        # ── Step 3: Evidence Selection & Deduplication ────────────────────────
        retrieval_meta = {
            "retriever": hybrid_result.retriever,
            "fusion_mode": hybrid_result.metadata.get("fusion_mode", "rrf"),
            "candidate_count": len(candidates),
        }
        ranking_meta = {
            "model_name": self.ranking_service.reranker.model_name,
            "device": self.ranking_service.reranker.device,
            "rerank_latency_ms": rerank_lat,
        }

        evidence_pack, evidence_lat = self.evidence_selector.select(
            query=request.query,
            reranked_candidates=reranked_chunks,
            max_evidence=request.evidence_k,
            min_score=request.min_score,
            retrieval_metadata=retrieval_meta,
            ranking_metadata=ranking_meta,
        )

        total_lat = round((time.perf_counter() - pipeline_start) * 1000, 2)

        latency_breakdown = LatencyBreakdown(
            bm25_latency_ms=bm25_lat,
            dense_latency_ms=dense_lat,
            hybrid_fusion_latency_ms=fusion_lat,
            rerank_latency_ms=rerank_lat,
            evidence_selection_latency_ms=evidence_lat,
            total_retrieval_latency_ms=total_lat,
        )

        logger.info(
            "research_retrieval_completed",
            query=request.query,
            candidates=len(candidates),
            reranked=len(reranked_chunks),
            evidence_selected=len(evidence_pack.selected_passages),
            total_latency_ms=total_lat,
        )

        return ResearchRetrievalResponse(
            query=request.query,
            total_candidates=len(candidates),
            total_evidence=len(evidence_pack.selected_passages),
            candidates=candidates,
            reranked=reranked_chunks,
            evidence_pack=evidence_pack,
            latency_breakdown=latency_breakdown,
        )

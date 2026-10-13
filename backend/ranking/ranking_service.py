"""
Dedicated ranking service for candidate passage scoring and ordering.
"""
from __future__ import annotations

import threading
import time
from typing import Any
import structlog

from core.config import Settings, get_settings

from ranking.base import Reranker
from ranking.cross_encoder_reranker import CrossEncoderReranker
from retrieval.models import ScoredChunk

logger = structlog.get_logger(__name__)


_shared_rerankers: dict[tuple[Any, ...], CrossEncoderReranker] = {}
_shared_lock = threading.Lock()


def get_shared_reranker(settings: Settings | None = None) -> CrossEncoderReranker:
    """
    Process-level CrossEncoderReranker, one per distinct model configuration.

    The model is loaded lazily once and reused by every request; the reranker itself
    serializes load and inference. Ranking semantics are unchanged.
    """
    st = settings or get_settings()
    key = (
        st.reranker_model,
        st.reranker_device,
        st.reranker_max_length,
        st.reranker_batch_size,
        st.reranker_fallback_mode,
    )
    with _shared_lock:
        inst = _shared_rerankers.get(key)
        if inst is None:
            inst = CrossEncoderReranker(settings=st)
            _shared_rerankers[key] = inst
        return inst


def reset_shared_rerankers() -> None:
    """Drop cached rerankers (tests / configuration changes)."""
    with _shared_lock:
        _shared_rerankers.clear()


class RankingService:
    """
    Orchestrates the reranking of candidate passages retrieved by hybrid search.
    """

    def __init__(self, reranker: Reranker | None = None) -> None:
        self.reranker = reranker or get_shared_reranker()

    async def rerank_candidates(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> tuple[list[ScoredChunk], float]:
        """
        Rerank hybrid candidate passages against the query.
        
        Returns:
            Tuple of (reranked_chunks, latency_ms)
        """
        if not candidates:
            return [], 0.0

        start_time = time.perf_counter()
        reranked = await self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return reranked, latency_ms

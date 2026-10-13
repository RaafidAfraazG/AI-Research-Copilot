"""
Cross-Encoder candidate passage reranker using Sentence Transformers / Hugging Face.
Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Literal
import structlog

from core.config import Settings, get_settings
from ranking.base import Reranker, RerankingError
from retrieval.models import ScoredChunk

logger = structlog.get_logger(__name__)


class CrossEncoderReranker(Reranker):
    """
    Passage reranker using a cross-encoder model to compute fine-grained
    query-passage interaction relevance scores.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
        fallback_mode: Literal["hybrid_score", "error"] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._model_name = model_name or self.settings.reranker_model
        self._device = device or self.settings.reranker_device
        self.max_length = max_length or self.settings.reranker_max_length
        self.batch_size = batch_size or self.settings.reranker_batch_size
        self.fallback_mode = fallback_mode or self.settings.reranker_fallback_mode

        # Lazy model instance
        self._model: Any = None
        self._load_failed: bool = False
        self._load_error_msg: str | None = None
        # Instances are shared process-wide (see ranking_service.get_shared_reranker):
        # serialize model load and inference so concurrent requests are safe and
        # scores stay deterministic.
        self._lock = threading.RLock()
        self._retry_at: float = 0.0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    # After a failed load the fallback is used; the load is retried after this cooldown
    # so a transient failure does not disable neural reranking for the whole process.
    _RETRY_COOLDOWN_S = 60.0

    def _load_model(self) -> Any:
        """
        Lazily load the cross-encoder model on first use (thread-safe, loaded once).
        """
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            if self._load_failed:
                if self._retry_at == 0.0 or time.monotonic() < self._retry_at:
                    if self.fallback_mode == "error":
                        raise RerankingError(
                            f"Cross-encoder model loading previously failed: {self._load_error_msg}"
                        )
                    return None
                self._load_failed = False  # cooldown elapsed: try again

            try:
                logger.info("loading_cross_encoder_model", model=self._model_name, device=self._device)
                from sentence_transformers import CrossEncoder  # type: ignore

                model = CrossEncoder(
                    model_name_or_path=self._model_name,
                    max_length=self.max_length,
                    device=self._device,
                )
                self._model = model
                logger.info("cross_encoder_model_loaded_successfully", model=self._model_name)
                return self._model
            except Exception as e:
                self._load_failed = True
                self._load_error_msg = type(e).__name__
                self._retry_at = time.monotonic() + self._RETRY_COOLDOWN_S
                logger.warning(
                    "cross_encoder_model_load_failed",
                    model=self._model_name,
                    error_type=type(e).__name__,
                    fallback_mode=self.fallback_mode,
                )
                if self.fallback_mode == "error":
                    raise RerankingError(
                        f"Failed to load cross-encoder model '{self._model_name}': {type(e).__name__}"
                    ) from e
                return None

    def warmup(self) -> bool:
        """
        Pre-load the model and execute one dummy inference to warm up the PyTorch runtime.
        
        Returns:
            True if model is loaded and ready for neural inference, False if fallback.
        """
        model = self._load_model()
        if model is None:
            return False
        try:
            self.score_pairs("warmup query", ["warmup passage"])
            logger.info("cross_encoder_warmed_up", model=self._model_name)
            return True
        except Exception as e:
            logger.warning("cross_encoder_warmup_failed", error=str(e))
            return False

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """
        Score a batch of (query, passage) pairs synchronously.
        """
        if not passages:
            return []

        model = self._load_model()
        if model is None:
            # Fallback mode active
            if self.fallback_mode == "error":
                raise RerankingError(f"Reranker unavailable: {self._load_error_msg}")
            # In fallback mode, assign uniform or index-decayed heuristic score
            return [0.0] * len(passages)

        pairs = [[query, text] for text in passages]
        try:
            with self._lock:
                raw_scores = model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
            return [float(s) for s in raw_scores]
        except Exception as e:
            logger.error("cross_encoder_prediction_failed", error=str(e))
            if self.fallback_mode == "error":
                raise RerankingError(f"Cross-encoder prediction failed: {e}") from e
            return [0.0] * len(passages)

    async def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """
        Rerank hybrid candidate passages using batch cross-encoder inference.
        """
        if not candidates:
            return []

        start_time = time.perf_counter()
        passages = [c.content for c in candidates]

        # Run CPU/GPU inference in threadpool to keep event loop unblocked
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self.score_pairs, query, passages)

        is_fallback = (self._model is None and not self.fallback_mode == "error" and all(s == 0.0 for s in scores))

        reranked_chunks: list[ScoredChunk] = []
        for i, candidate in enumerate(candidates):
            score = scores[i] if (not is_fallback or scores[i] != 0.0) else (candidate.score or 0.0)
            
            chunk_copy = candidate.model_copy()
            chunk_copy.rerank_score = score
            chunk_copy.score = score
            chunk_copy.retriever_name = "cross_encoder"
            
            meta = chunk_copy.metadata.copy()
            meta["original_retriever"] = candidate.retriever_name
            meta["original_rank"] = candidate.rank
            meta["original_score"] = candidate.score
            if is_fallback:
                meta["reranker_fallback"] = True
            chunk_copy.metadata = meta
            
            reranked_chunks.append(chunk_copy)

        # Sort descending by rerank score
        reranked_chunks.sort(key=lambda c: (c.rerank_score if c.rerank_score is not None else float("-inf")), reverse=True)

        # Re-assign 1-indexed ranks
        for rank_idx, chunk in enumerate(reranked_chunks, start=1):
            chunk.rank = rank_idx


        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "reranking_completed",
            query=query,
            candidates_count=len(candidates),
            model=self._model_name,
            latency_ms=elapsed_ms,
            fallback=is_fallback,
        )

        if top_k is not None and top_k > 0:
            return reranked_chunks[:top_k]
        return reranked_chunks

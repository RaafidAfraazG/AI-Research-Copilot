"""
Abstract base class and exceptions for candidate passage rerankers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from retrieval.models import ScoredChunk


class RerankingError(Exception):
    """Raised when a reranker encounters an unrecoverable failure."""
    pass


class Reranker(ABC):
    """
    Abstract base class for model-agnostic passage reranking.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name or identifier of the underlying ranking model."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the compute device (e.g. 'cpu', 'cuda')."""
        pass

    @abstractmethod
    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """
        Compute relevance scores for a batch of (query, passage) pairs.
        
        Args:
            query: The user research query.
            passages: List of candidate passage text strings.
            
        Returns:
            List of float relevance scores in the same order as input passages.
        """
        pass

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """
        Rerank a list of candidate ScoredChunk instances against a query.
        
        Args:
            query: The search query.
            candidates: Candidate chunks retrieved from hybrid search.
            top_k: Optional maximum number of top reranked chunks to return.
            
        Returns:
            Sorted list of ScoredChunks with updated rerank_score, score, and rank.
        """
        pass

    def warmup(self) -> bool:
        """
        Optional warmup method to pre-load model weights and perform dummy inference.
        
        Returns:
            True if warmup succeeded or is a no-op, False if model failed to load.
        """
        return True

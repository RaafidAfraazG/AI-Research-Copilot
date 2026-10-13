"""
Information retrieval and ranking evaluation metrics: Recall@K, Precision@K, MRR, NDCG@K,
and statistical aggregation helpers.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Sequence
from pydantic import BaseModel, Field


class DistributionStats(BaseModel):
    """Statistical summary of a metric distribution."""
    mean: float = 0.0
    median: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    std_dev: float = 0.0
    count: int = 0


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """
    Compute Recall@K.
    Proportion of unique relevant documents successfully retrieved within top-K.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    retrieved_relevant = set(top_k).intersection(relevant_ids)
    return min(1.0, len(retrieved_relevant) / len(relevant_ids))


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """
    Compute Precision@K.
    Proportion of retrieved documents in top-K that are relevant.
    """
    if not retrieved_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    actual_k = len(top_k)
    if actual_k == 0:
        return 0.0

    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / actual_k


def mrr(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    Reciprocal rank of the first relevant document in the retrieved list (1/rank).
    """
    if not relevant_ids:
        return 0.0

    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    # Compute DCG (unique relevant hits)
    dcg = 0.0
    seen_relevant: set[str] = set()
    for idx, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids and doc_id not in seen_relevant:
            seen_relevant.add(doc_id)
            dcg += 1.0 / math.log2(idx + 1)

    # Compute Ideal DCG (IDCG)
    idcg = 0.0
    ideal_hits = min(len(relevant_ids), k)
    for idx in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(idx + 1)

    if idcg == 0.0:
        return 0.0
    return min(1.0, dcg / idcg)


def compute_distribution_stats(values: list[float]) -> DistributionStats:
    """
    Compute summary statistics: mean, median, min, max, std_dev, and count.
    """
    if not values:
        return DistributionStats(mean=0.0, median=0.0, min_val=0.0, max_val=0.0, std_dev=0.0, count=0)

    mean_val = statistics.mean(values)
    median_val = statistics.median(values)
    min_val = min(values)
    max_val = max(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0.0

    return DistributionStats(
        mean=round(mean_val, 4),
        median=round(median_val, 4),
        min_val=round(min_val, 4),
        max_val=round(max_val, 4),
        std_dev=round(std_val, 4),
        count=len(values),
    )

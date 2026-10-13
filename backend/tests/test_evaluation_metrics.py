"""
Unit tests for information retrieval metrics (Recall@K, MRR, NDCG@K).
"""
from __future__ import annotations

import pytest
from evaluation.metrics import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k():
    relevant = {"doc-1", "doc-2"}
    
    # Hits at rank 1 and 2
    retrieved = ["doc-1", "doc-2", "doc-3", "doc-4"]
    assert recall_at_k(retrieved, relevant, k=1) == 0.5
    assert recall_at_k(retrieved, relevant, k=2) == 1.0
    assert recall_at_k(retrieved, relevant, k=5) == 1.0

    # No hits in top-1
    retrieved_late = ["doc-3", "doc-1", "doc-4"]
    assert recall_at_k(retrieved_late, relevant, k=1) == 0.0
    assert recall_at_k(retrieved_late, relevant, k=2) == 0.5


def test_mrr():
    relevant = {"doc-target"}

    # Hit at rank 1 -> MRR 1.0
    assert mrr(["doc-target", "doc-2"], relevant) == 1.0

    # Hit at rank 2 -> MRR 0.5
    assert mrr(["doc-other", "doc-target"], relevant) == 0.5

    # Hit at rank 4 -> MRR 0.25
    assert mrr(["doc-1", "doc-2", "doc-3", "doc-target"], relevant) == 0.25

    # No hits -> MRR 0.0
    assert mrr(["doc-1", "doc-2"], relevant) == 0.0


def test_ndcg_at_k():
    relevant = {"doc-1", "doc-2"}

    # Perfect ranking: hits at rank 1 and 2
    perfect = ["doc-1", "doc-2", "doc-3"]
    assert pytest.approx(ndcg_at_k(perfect, relevant, k=3), rel=1e-3) == 1.0

    # Imperfect ranking: hits at rank 2 and 3
    suboptimal = ["doc-3", "doc-1", "doc-2"]
    ndcg_val = ndcg_at_k(suboptimal, relevant, k=3)
    assert 0.0 < ndcg_val < 1.0

    # Zero hits
    miss = ["doc-3", "doc-4"]
    assert ndcg_at_k(miss, relevant, k=2) == 0.0

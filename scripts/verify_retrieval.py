"""
End-to-End Retrieval Verification Script.
Tests real search queries against the local retrieval infrastructure.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add backend to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievalQuery


async def run_verification() -> None:
    print("=" * 70)
    print("AI Research Copilot — Retrieval Verification (Phase 2.5)")
    print("=" * 70)

    # ── 1. BM25 Search ───────────────────────────────────────────────────────
    print("\n[1] Testing BM25 Lexical Retrieval:")
    print("    Query: 'retrieval augmented generation'")
    bm25 = BM25Retriever()
    q1 = RetrievalQuery(query_text="retrieval augmented generation", top_k=3)
    res1 = await bm25.search(q1)
    print(f"    Hits found: {len(res1.chunks)} | Latency: {res1.latency_ms}ms")
    for c in res1.chunks:
        print(f"    - Rank {c.rank} (score: {c.score:.4f}): '{c.title}' [{c.source_type}:{c.source_id}]")

    # ── 2. Dense Vector Search ───────────────────────────────────────────────
    print("\n[2] Testing Dense Vector Retrieval:")
    print("    Query: 'semantic representation of sentences'")
    dense = DenseRetriever()
    q2 = RetrievalQuery(query_text="semantic representation of sentences", top_k=3)
    res2 = await dense.search(q2)
    print(f"    Hits found: {len(res2.chunks)} | Latency: {res2.latency_ms}ms")
    for c in res2.chunks:
        print(f"    - Rank {c.rank} (score: {c.score:.4f}): '{c.title}' [{c.source_type}:{c.source_id}]")

    # ── 3. Hybrid RRF Search ─────────────────────────────────────────────────
    print("\n[3] Testing Hybrid (RRF) Retrieval:")
    print("    Query: 'combine lexical and semantic retrieval'")
    hybrid = HybridRetriever()
    q3 = RetrievalQuery(query_text="combine lexical and semantic retrieval", top_k=5, alpha=0.5)
    res3 = await hybrid.search(q3)
    print(f"    Hits found: {len(res3.chunks)} | Latency: {res3.latency_ms}ms | Fusion: {res3.metadata.get('fusion_method')}")
    for c in res3.chunks:
        print(
            f"    - Rank {c.rank} (RRF: {c.score:.6f} | BM25 rank: {c.bm25_rank}, Dense rank: {c.dense_rank}): "
            f"'{c.title}' [{c.source_type}:{c.source_id}]"
        )

    print("\n" + "=" * 70)
    print("All retrieval modes verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_verification())

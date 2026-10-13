"""
CLI script to verify Candidate Retrieval -> Cross-Encoder Reranking -> Evidence Selection.

Usage:
    python scripts/verify_reranking.py "How does hybrid retrieval improve RAG systems?"
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

# Add backend to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from ranking.evidence import EvidencePack
from services.research_retrieval_service import (
    ResearchRetrievalRequest,
    ResearchRetrievalService,
)



async def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Reranking & Evidence Selection Pipeline")
    parser.add_argument(
        "query",
        nargs="?",
        default="How does hybrid retrieval improve RAG systems?",
        help="Research query string",
    )
    parser.add_argument("--candidate-k", type=int, default=25, help="Number of hybrid candidates")
    parser.add_argument("--evidence-k", type=int, default=6, help="Target number of evidence passages")
    args = parser.parse_args()

    print("=" * 75)
    print(f"Research Query: '{args.query}'")
    print(f"Candidate K: {args.candidate_k} | Evidence Target K: {args.evidence_k}")
    print("=" * 75)

    service = ResearchRetrievalService()
    req = ResearchRetrievalRequest(
        query=args.query,
        candidate_k=args.candidate_k,
        evidence_k=args.evidence_k,
    )

    resp = await service.retrieve_evidence(req)

    print("\n[1] HYBRID CANDIDATE RETRIEVAL (Top 5 of " + str(resp.total_candidates) + "):")
    print("-" * 75)
    for c in resp.candidates[:5]:
        print(f"  - Rank {c.rank} (Score: {c.score:.4f} | Source: {c.source_type}:{c.source_id})")
        print(f"    Title: {c.title}")
        print(f"    Text:  {c.content[:100]}...")

    print("\n[2] CROSS-ENCODER RERANKED PASSAGES (Top 5):")
    print("-" * 75)
    for r in resp.reranked[:5]:
        print(f"  - Rank {r.rank} (Rerank Score: {r.rerank_score:.4f} | Prev Rank: {r.metadata.get('original_rank')})")
        print(f"    Title: {r.title} [{r.source_id}]")
        print(f"    Text:  {r.content[:100]}...")

    print("\n[3] FINAL ASSEMBLED EVIDENCE PACK (Selected: " + str(resp.total_evidence) + "):")
    print("-" * 75)
    pack: EvidencePack = resp.evidence_pack
    for ep in pack.selected_passages:
        print(f"  [Evidence #{ep.rank}] Score: {ep.relevance_score:.4f} | Doc: {ep.document_id[:8]}... | Source: {ep.source_type}:{ep.source_id}")
        print(f"    Title: {ep.title}")
        if ep.url:
            print(f"    URL:   {ep.url}")
        print(f"    Text:  {ep.text[:140]}...")
        print()

    print("[4] PIPELINE LATENCY BREAKDOWN:")
    print("-" * 75)
    lb = resp.latency_breakdown
    print(f"  - BM25 Lexical Latency:     {lb.bm25_latency_ms:>8.2f} ms")
    print(f"  - Dense Vector Latency:     {lb.dense_latency_ms:>8.2f} ms")
    print(f"  - Hybrid Fusion Latency:    {lb.hybrid_fusion_latency_ms:>8.2f} ms")
    print(f"  - Cross-Encoder Rerank:     {lb.rerank_latency_ms:>8.2f} ms")
    print(f"  - Evidence Selection:       {lb.evidence_selection_latency_ms:>8.2f} ms")
    print(f"  - Total Pipeline Latency:   {lb.total_retrieval_latency_ms:>8.2f} ms")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())

"""
CLI script to run retrieval and reranking evaluation benchmark.

Usage:
    python scripts/evaluate_retrieval.py
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys

# Add backend to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from evaluation.evaluator import RetrievalEvaluator



async def main() -> None:
    print("=" * 80)
    print("AI Research Copilot — Retrieval & Reranking Benchmark Evaluation")
    print("Comparing: Hybrid Retrieval (BM25 + Dense) vs. Hybrid + Cross-Encoder Reranking")
    print("=" * 80)

    evaluator = RetrievalEvaluator()
    report = await evaluator.evaluate()

    print(f"\nEvaluated {report.total_queries} Benchmark Queries in {report.elapsed_seconds:.2f}s:\n")

    header = f"{'Query ID':<10} | {'Mode':<18} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<8} | {'NDCG@10':<8}"
    print(header)
    print("-" * 80)

    for q in report.query_results:
        print(f"{q.query_id:<10} | {'Hybrid (Base)':<18} | {q.hybrid_metrics.recall_at_5:<10.4f} | {q.hybrid_metrics.recall_at_10:<10.4f} | {q.hybrid_metrics.mrr:<8.4f} | {q.hybrid_metrics.ndcg_at_10:<8.4f}")
        print(f"{'':<10} | {'+ Cross-Encoder':<18} | {q.reranked_metrics.recall_at_5:<10.4f} | {q.reranked_metrics.recall_at_10:<10.4f} | {q.reranked_metrics.mrr:<8.4f} | {q.reranked_metrics.ndcg_at_10:<8.4f}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("SUMMARY METRICS (MEAN ACROSS ALL QUERIES):")
    print("=" * 80)
    hm = report.hybrid_mean_metrics
    rm = report.reranked_mean_metrics

    print(f"{'Metric':<25} | {'Hybrid Retrieval':<18} | {'Hybrid + Cross-Encoder':<22} | {'Delta':<10}")
    print("-" * 80)
    print(f"{'Mean Recall@5':<25} | {hm.recall_at_5:<18.4f} | {rm.recall_at_5:<22.4f} | {rm.recall_at_5 - hm.recall_at_5:+.4f}")
    print(f"{'Mean Recall@10':<25} | {hm.recall_at_10:<18.4f} | {rm.recall_at_10:<22.4f} | {rm.recall_at_10 - hm.recall_at_10:+.4f}")
    print(f"{'Mean MRR':<25} | {hm.mrr:<18.4f} | {rm.mrr:<22.4f} | {rm.mrr - hm.mrr:+.4f}")
    print(f"{'Mean NDCG@10':<25} | {hm.ndcg_at_10:<18.4f} | {rm.ndcg_at_10:<22.4f} | {rm.ndcg_at_10 - hm.ndcg_at_10:+.4f}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

"""
CLI Runner for the Comprehensive System Evaluation Benchmark (Phase 7).
Evaluates Retrieval, Reranking, Decomposition, Grounded Synthesis, Citation Integrity, and Latency.

Usage:
    python scripts/evaluate_system.py
    python scripts/evaluate_system.py --retrieval-only
    python scripts/evaluate_system.py --answers
    python scripts/evaluate_system.py --variant hybrid --variant full
    python scripts/evaluate_system.py --output reports/evaluation/
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Add backend directory to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from evaluation.system_evaluator import SystemEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 7 Retrieval, Grounding & Answer Quality Evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/evaluation/retrieval_benchmark.json",
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--variant",
        "-v",
        action="append",
        dest="variants",
        help="Specific variant(s) to evaluate (e.g. bm25, dense, hybrid, rerank, decomp, full)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run retrieval variants only (A: BM25, B: Dense, C: Hybrid, D: Hybrid+Rerank, E: Decomp+Hybrid)",
    )
    parser.add_argument(
        "--answers",
        action="store_true",
        help="Run full answer generation and grounding verification (Variant F)",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=25,
        help="Candidate retrieval pool size (default: 25)",
    )
    parser.add_argument(
        "--evidence-k",
        type=int,
        default=6,
        help="Target evidence count (default: 6)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="reports/evaluation",
        help="Output directory for JSON and Markdown reports",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write report files to disk (print to console only)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Determine variants to evaluate
    variants_to_run: list[str] = []
    if args.retrieval_only:
        variants_to_run = ["a", "b", "c", "d", "e"]
    elif args.answers:
        variants_to_run = ["f"]
    elif args.variants:
        for v in args.variants:
            variants_to_run.extend([x.strip().lower() for x in v.split(",") if x.strip()])
    else:
        variants_to_run = ["a", "b", "c", "d", "e", "f"]

    print("=" * 85)
    print("  AI RESEARCH COPILOT — SYSTEM BENCHMARK & EVALUATION ENGINE (PHASE 7)")
    print("=" * 85)
    print(f"Dataset:       {args.dataset}")
    print(f"Variants:      {', '.join(variants_to_run)}")
    print(f"Candidate K:   {args.candidate_k}")
    print(f"Evidence K:    {args.evidence_k}")
    print("=" * 85)

    evaluator = SystemEvaluator()
    dataset = evaluator.load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} benchmark queries.")

    reranker_mode, is_neural = evaluator.inspect_reranker_status()
    llm_name, is_real_llm = evaluator.inspect_llm_status()

    print(f"Reranker Status: {reranker_mode} (Neural: {is_neural})")
    print(f"LLM Status:      {llm_name} (Real Model: {is_real_llm})\n")

    print("Executing benchmark across requested variants...")
    report = await evaluator.run_full_benchmark(
        dataset=dataset,
        variants_to_run=variants_to_run,
        candidate_k=args.candidate_k,
        evidence_k=args.evidence_k,
    )

    # Render Markdown representation
    md_content = evaluator.render_markdown_report(report)

    # Print summary table to console
    print("\n" + "=" * 85)
    print("  RETRIEVAL & ABLATION RESULTS SUMMARY")
    print("=" * 85)
    header = f"{'Variant':<34} | {'Recall@5':<9} | {'Recall@10':<9} | {'MRR':<7} | {'NDCG@10':<8} | {'Latency (ms)':<12}"
    print(header)
    print("-" * 85)
    for row in report.ablation_summary:
        v_name = row["variant_name"]
        if len(v_name) > 34:
            v_name = v_name[:31] + "..."
        r5 = f"{row['recall_at_5']:.4f}"
        r10 = f"{row['recall_at_10']:.4f}"
        mrr_val = f"{row['mrr']:.4f}"
        ndcg10 = f"{row['ndcg_at_10']:.4f}"
        lat = f"{row['avg_latency_ms']:.2f}"
        print(f"{v_name:<34} | {r5:<9} | {r10:<9} | {mrr_val:<7} | {ndcg10:<8} | {lat:<12}")
    print("=" * 85)

    # If Answer / Full pipeline evaluated, print answer metrics
    if "variant_f_full_pipeline" in report.variants:
        vf = report.variants["variant_f_full_pipeline"]
        print("\n" + "=" * 85)
        print("  GROUNDED SYNTHESIS & CITATION EVALUATION (FULL PIPELINE)")
        print("=" * 85)
        for k, v in vf.mean_answer_metrics.items():
            print(f"  * {k:<30}: {v:.4f}")
        print("=" * 85)

    # Save reports
    if not args.no_save:
        out_dir = _ROOT / args.output
        out_dir.mkdir(parents=True, exist_ok=True)

        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        latest_json_path = out_dir / "latest.json"
        latest_md_path = out_dir / "latest.md"
        ts_json_path = out_dir / f"evaluation_{ts_str}.json"
        ts_md_path = out_dir / f"evaluation_{ts_str}.md"

        json_str = report.model_dump_json(indent=2)

        # Write latest
        latest_json_path.write_text(json_str, encoding="utf-8")
        latest_md_path.write_text(md_content, encoding="utf-8")

        # Write timestamped archival copies
        ts_json_path.write_text(json_str, encoding="utf-8")
        ts_md_path.write_text(md_content, encoding="utf-8")

        print(f"\nReports saved successfully:")
        print(f"  - Latest JSON: {latest_json_path}")
        print(f"  - Latest MD:   {latest_md_path}")
        print(f"  - Timestamped: {ts_json_path}")


if __name__ == "__main__":
    asyncio.run(main())

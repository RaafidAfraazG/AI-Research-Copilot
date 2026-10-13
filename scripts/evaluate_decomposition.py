"""
Script to evaluate Query Intelligence and Query Decomposition on the benchmark dataset.
Usage:
    python scripts/evaluate_decomposition.py
"""
import asyncio
from pathlib import Path
import sys

# Add backend to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from evaluation.decomposition_evaluator import DecompositionEvaluator


async def main() -> None:
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "query_decomposition_benchmark.json"
    print(f"Loading query decomposition benchmark: {dataset_path}")

    evaluator = DecompositionEvaluator()
    report = await evaluator.evaluate_dataset(dataset_path, mode="deterministic")

    print("\n" + "=" * 70)
    print("QUERY DECOMPOSITION BENCHMARK REPORT")
    print("=" * 70)
    print(f"Total Queries Evaluated:          {report.total_queries}")
    print(f"Decomposition Success Rate:      {report.decomposition_success_rate * 100:.1f}%")
    print(f"Subquery Coverage:               {report.subquery_coverage * 100:.1f}%")
    print(f"Redundancy Rate:                 {report.redundancy_rate * 100:.1f}%")
    print(f"Constraint Preservation:         {report.constraint_preservation * 100:.1f}%")
    print(f"Intent Classification Accuracy:  {report.intent_accuracy * 100:.1f}%")
    print(f"Complexity Accuracy:             {report.complexity_accuracy * 100:.1f}%")
    print(f"Decomposition Decision Accuracy: {report.decomposition_decision_accuracy * 100:.1f}%")
    print(f"Evaluation Methodology:          {report.evaluation_type}")
    print("=" * 70)

    print("\nPer-Query Detailed Results:")
    for d in report.details:
        status_tag = "[PASS]" if d.success and d.subquery_coverage >= 0.75 else "[WARN]"
        print(f"{status_tag} {d.query_id}: {d.query[:60]}...")
        print(f"    Intent: {d.predicted_intent} (Expected: {d.expected_intent}) | Match: {d.intent_match}")
        print(f"    Complexity: {d.predicted_complexity} (Decomp: {d.requires_decomposition_predicted}) | Subqueries: {d.subquery_count}")
        print(f"    Coverage: {d.subquery_coverage * 100:.1f}% | Redundancy: {d.redundancy_rate * 100:.1f}% | Preservation: {d.constraint_preservation * 100:.1f}%")
        for idx, sq in enumerate(d.subqueries, 1):
            print(f"      sq_{idx:03d}: {sq}")
        print()


if __name__ == "__main__":
    asyncio.run(main())

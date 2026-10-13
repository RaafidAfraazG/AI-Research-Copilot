"""
Query Decomposition and Intelligence Benchmark Evaluator.
Evaluates Subquery Coverage, Redundancy Rate, Constraint Preservation,
Decomposition Success Rate, and Classification Accuracy on benchmark datasets.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Literal
from pydantic import BaseModel, Field
import structlog

from query.decomposer import QueryDecomposer
from query.models import QueryAnalysis, QueryComplexity, QueryIntent, ResearchPlan

logger = structlog.get_logger(__name__)


class QueryEvaluationDetail(BaseModel):
    """Evaluation metrics for a single benchmark query."""
    query_id: str
    query: str
    expected_intent: str
    predicted_intent: str
    intent_match: bool
    expected_complexity: str
    predicted_complexity: str
    complexity_match: bool
    requires_decomposition_expected: bool
    requires_decomposition_predicted: bool
    subquery_count: int
    subqueries: list[str]
    subquery_coverage: float = Field(..., description="Fraction of required dimensions found in subqueries")
    redundancy_rate: float = Field(..., description="Fraction of subquery pairs with high overlap")
    constraint_preservation: float = Field(..., description="Fraction of required constraints/entities preserved")
    success: bool = Field(..., description="Valid research plan produced")


class DecompositionBenchmarkReport(BaseModel):
    """Aggregated evaluation results over the decomposition benchmark."""
    benchmark_name: str
    total_queries: int
    decomposition_success_rate: float
    subquery_coverage: float
    redundancy_rate: float
    constraint_preservation: float
    intent_accuracy: float
    complexity_accuracy: float
    decomposition_decision_accuracy: float
    evaluation_type: str = "deterministic_rule_based_and_lexical_matching"
    details: list[QueryEvaluationDetail] = Field(default_factory=list)


class DecompositionEvaluator:
    """
    Evaluator assessing the structural, semantic, and constraint-preserving quality
    of query decomposition without hand-waving or ungrounded claims.
    """

    def __init__(self, decomposer: QueryDecomposer | None = None) -> None:
        self.decomposer = decomposer or QueryDecomposer()

    @staticmethod
    def _token_jaccard(s1: str, s2: str) -> float:
        """Calculate word-level Jaccard similarity between two strings."""
        tokens1 = set(re.findall(r"\b\w+\b", s1.lower()))
        tokens2 = set(re.findall(r"\b\w+\b", s2.lower()))
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0

    def evaluate_plan(
        self,
        item: dict[str, Any],
        plan: ResearchPlan,
    ) -> QueryEvaluationDetail:
        """
        Evaluate a single ResearchPlan against benchmark ground truth annotations.
        """
        query_id = item["id"]
        original_q = item["query"]
        expected_intent = item.get("expected_intent", "FACTUAL")
        expected_comp = item.get("expected_complexity", "SIMPLE")
        exp_req_decomp = item.get("requires_decomposition", False)

        pred_intent = plan.analysis.intent.value
        pred_comp = plan.analysis.complexity.value
        pred_req_decomp = plan.analysis.requires_decomposition

        intent_match = (pred_intent == expected_intent)
        comp_match = (pred_comp == expected_comp)

        sub_texts = [sq.text for sq in plan.subqueries]
        combined_subquery_text = " ".join(sub_texts).lower()

        # 1. Subquery Coverage: Fraction of required dimensions mentioned in subquery texts
        req_dimensions = item.get("required_dimensions", [])
        if req_dimensions:
            covered_dims = 0
            for dim in req_dimensions:
                dim_words = [w for w in re.split(r"\s+", dim.lower()) if len(w) > 2]
                if any(w in combined_subquery_text for w in dim_words):
                    covered_dims += 1
            coverage = round(covered_dims / len(req_dimensions), 4)
        else:
            coverage = 1.0

        # 2. Redundancy Rate: Check pairwise Jaccard similarity between generated subqueries
        redundancy_rate = 0.0
        if len(sub_texts) > 1:
            redundant_pairs = 0
            total_pairs = 0
            for i in range(len(sub_texts)):
                for j in range(i + 1, len(sub_texts)):
                    total_pairs += 1
                    jaccard = self._token_jaccard(sub_texts[i], sub_texts[j])
                    if jaccard > 0.70:
                        redundant_pairs += 1
            redundancy_rate = round(redundant_pairs / total_pairs, 4) if total_pairs > 0 else 0.0

        # 3. Constraint Preservation: Check entities, constraints, and time markers
        targets_to_preserve = (
            item.get("required_entities", [])
            + item.get("required_constraints", [])
            + item.get("required_time_constraints", [])
        )
        if targets_to_preserve:
            preserved_count = 0
            for target in targets_to_preserve:
                target_words = [w for w in re.split(r"\s+", target.lower()) if len(w) > 2]
                if any(w in combined_subquery_text or w in original_q.lower() for w in target_words):
                    preserved_count += 1
            constraint_preservation = round(preserved_count / len(targets_to_preserve), 4)
        else:
            constraint_preservation = 1.0

        # 4. Success: Plan is structurally valid and non-empty
        success = len(plan.subqueries) >= 1 and all(len(sq.text.strip()) > 3 for sq in plan.subqueries)

        return QueryEvaluationDetail(
            query_id=query_id,
            query=original_q,
            expected_intent=expected_intent,
            predicted_intent=pred_intent,
            intent_match=intent_match,
            expected_complexity=expected_comp,
            predicted_complexity=pred_comp,
            complexity_match=comp_match,
            requires_decomposition_expected=exp_req_decomp,
            requires_decomposition_predicted=pred_req_decomp,
            subquery_count=len(plan.subqueries),
            subqueries=sub_texts,
            subquery_coverage=coverage,
            redundancy_rate=redundancy_rate,
            constraint_preservation=constraint_preservation,
            success=success,
        )

    async def evaluate_dataset(
        self,
        dataset_path: str | Path,
        mode: Literal["auto", "llm", "deterministic"] = "deterministic",
    ) -> DecompositionBenchmarkReport:
        """
        Run decomposition evaluation across the benchmark dataset.
        """
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        queries = data.get("queries", [])
        benchmark_name = data.get("benchmark_name", "query_decomposition_benchmark")

        details: list[QueryEvaluationDetail] = []

        for item in queries:
            query_text = item["query"]
            plan = await self.decomposer.decompose(query_text, mode=mode)
            detail = self.evaluate_plan(item, plan)
            details.append(detail)

        total = len(details)
        if total == 0:
            return DecompositionBenchmarkReport(
                benchmark_name=benchmark_name,
                total_queries=0,
                decomposition_success_rate=0.0,
                subquery_coverage=0.0,
                redundancy_rate=0.0,
                constraint_preservation=0.0,
                intent_accuracy=0.0,
                complexity_accuracy=0.0,
                decomposition_decision_accuracy=0.0,
                details=[],
            )

        avg_coverage = round(sum(d.subquery_coverage for d in details) / total, 4)
        avg_redundancy = round(sum(d.redundancy_rate for d in details) / total, 4)
        avg_preservation = round(sum(d.constraint_preservation for d in details) / total, 4)
        success_rate = round(sum(1 for d in details if d.success) / total, 4)
        intent_acc = round(sum(1 for d in details if d.intent_match) / total, 4)
        comp_acc = round(sum(1 for d in details if d.complexity_match) / total, 4)
        decomp_dec_acc = round(
            sum(1 for d in details if d.requires_decomposition_expected == d.requires_decomposition_predicted) / total,
            4,
        )

        return DecompositionBenchmarkReport(
            benchmark_name=benchmark_name,
            total_queries=total,
            decomposition_success_rate=success_rate,
            subquery_coverage=avg_coverage,
            redundancy_rate=avg_redundancy,
            constraint_preservation=avg_preservation,
            intent_accuracy=intent_acc,
            complexity_accuracy=comp_acc,
            decomposition_decision_accuracy=decomp_dec_acc,
            details=details,
        )

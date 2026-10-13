"""
Evaluation framework for retrieval, ranking, grounding, and answer quality.
"""
from evaluation.answer_metrics import AnswerEvaluationScore, AnswerMetricsEvaluator
from evaluation.dataset import RESEARCH_EVALUATION_DATASET, EvaluationQuery
from evaluation.evaluator import EvaluationReport, QueryEvaluationResult, RetrievalEvaluator
from evaluation.metrics import (
    DistributionStats,
    compute_distribution_stats,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from evaluation.system_evaluator import (
    BenchmarkQueryItem,
    SingleQueryEvaluation,
    SystemBenchmarkReport,
    SystemEvaluator,
    VariantEvaluationResult,
)

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "compute_distribution_stats",
    "DistributionStats",
    "EvaluationQuery",
    "RESEARCH_EVALUATION_DATASET",
    "RetrievalEvaluator",
    "EvaluationReport",
    "QueryEvaluationResult",
    "AnswerMetricsEvaluator",
    "AnswerEvaluationScore",
    "SystemEvaluator",
    "SystemBenchmarkReport",
    "VariantEvaluationResult",
    "BenchmarkQueryItem",
    "SingleQueryEvaluation",
]

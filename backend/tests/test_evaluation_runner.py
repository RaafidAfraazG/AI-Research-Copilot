"""
Tests for SystemEvaluator multi-variant benchmark execution, report generation, and status detection (Phase 7).
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from evaluation.system_evaluator import (
    BenchmarkQueryItem,
    SystemBenchmarkReport,
    SystemEvaluator,
    VariantEvaluationResult,
)

# Retrieval runs against an isolated index built with the EXPLICITLY configured mock
# embedder (conftest.mock_index); these tests never touch the real repository index.
pytestmark = pytest.mark.usefixtures("mock_index")


@pytest.fixture
def sample_dataset() -> list[BenchmarkQueryItem]:
    return [
        BenchmarkQueryItem(
            query_id="test_01",
            query="What is Retrieval-Augmented Generation for NLP?",
            intent="FACTUAL",
            difficulty="simple",
            requires_decomposition=False,
            relevant_documents=["2005.11401", "arxiv:2005.11401"],
            required_dimensions=["retrieval-augmented generation", "parametric memory"],
            expected_dimensions=["parametric memory", "dense retriever"],
            ground_truth_source="Lewis et al., 2020",
        ),
        BenchmarkQueryItem(
            query_id="test_02",
            query="How does Dense Passage Retrieval compare with BM25 keyword matching?",
            intent="COMPARATIVE",
            difficulty="moderate",
            requires_decomposition=True,
            relevant_documents=["2004.04906", "arxiv:2004.04906"],
            required_dimensions=["dense passage retrieval", "bm25"],
            expected_dimensions=["dense passage retrieval", "bm25 keyword matching"],
            ground_truth_source="Karpukhin et al., 2020",
        ),
    ]


@pytest.mark.asyncio
async def test_system_evaluator_variant_a_bm25(sample_dataset: list[BenchmarkQueryItem]) -> None:
    evaluator = SystemEvaluator()
    res: VariantEvaluationResult = await evaluator.evaluate_variant_a_bm25(sample_dataset, candidate_k=10)

    assert res.variant_id == "variant_a_bm25"
    assert res.total_queries == 2
    assert res.executed_queries == 2
    assert "recall_at_5" in res.mean_retrieval_metrics
    assert "mrr" in res.mean_retrieval_metrics
    assert "ndcg_at_10" in res.mean_retrieval_metrics
    assert len(res.query_evaluations) == 2


@pytest.mark.asyncio
async def test_system_evaluator_variant_b_dense(sample_dataset: list[BenchmarkQueryItem]) -> None:
    evaluator = SystemEvaluator()
    res: VariantEvaluationResult = await evaluator.evaluate_variant_b_dense(sample_dataset, candidate_k=10)

    assert res.variant_id == "variant_b_dense"
    assert res.total_queries == 2
    assert len(res.query_evaluations) == 2


@pytest.mark.asyncio
async def test_system_evaluator_variant_c_hybrid(sample_dataset: list[BenchmarkQueryItem]) -> None:
    evaluator = SystemEvaluator()
    res: VariantEvaluationResult = await evaluator.evaluate_variant_c_hybrid(sample_dataset, candidate_k=10)

    assert res.variant_id == "variant_c_hybrid"
    assert res.total_queries == 2
    assert len(res.query_evaluations) == 2


@pytest.mark.asyncio
async def test_system_evaluator_variant_d_hybrid_reranker(sample_dataset: list[BenchmarkQueryItem]) -> None:
    evaluator = SystemEvaluator()
    res: VariantEvaluationResult = await evaluator.evaluate_variant_d_hybrid_rerank(sample_dataset, candidate_k=10)

    assert res.variant_id == "variant_d_hybrid_reranker"
    assert res.total_queries == 2
    assert res.reranker_mode in {"none", "fallback_active (hybrid_score)", "neural_loaded"} or "loaded" in res.reranker_mode or "fallback" in res.reranker_mode


@pytest.mark.asyncio
async def test_system_evaluator_variant_e_decomp_hybrid(sample_dataset: list[BenchmarkQueryItem]) -> None:
    evaluator = SystemEvaluator()
    res: VariantEvaluationResult = await evaluator.evaluate_variant_e_decomp_hybrid(sample_dataset, candidate_k=10)

    assert res.variant_id == "variant_e_decomp_hybrid"
    assert res.total_queries == 2
    assert len(res.query_evaluations) == 2


@pytest.mark.asyncio
async def test_system_evaluator_variant_f_full_pipeline(sample_dataset: list[BenchmarkQueryItem]) -> None:
    evaluator = SystemEvaluator()
    res: VariantEvaluationResult = await evaluator.evaluate_variant_f_full_pipeline(sample_dataset, candidate_k=10)

    assert res.variant_id == "variant_f_full_pipeline"
    assert res.total_queries == 2
    assert "claim_support_rate" in res.mean_answer_metrics
    assert "citation_coverage" in res.mean_answer_metrics
    assert "citation_validity_rate" in res.mean_answer_metrics


@pytest.mark.asyncio
async def test_system_evaluator_run_full_benchmark_and_markdown(sample_dataset: list[BenchmarkQueryItem]) -> None:
    evaluator = SystemEvaluator()
    report: SystemBenchmarkReport = await evaluator.run_full_benchmark(
        dataset=sample_dataset,
        variants_to_run=["a", "c", "f"],
        candidate_k=10,
    )

    assert report.benchmark_name == "ai_research_copilot_system_benchmark"
    assert len(report.variants) == 3
    assert len(report.ablation_summary) == 3

    md = evaluator.render_markdown_report(report)
    assert "# AI Research Copilot — Evaluation & Ablation Report" in md
    assert "System Retrieval Ablation Study" in md
    assert "Variant A: BM25" in md
    assert "Variant F: Full Research Pipeline" in md

    # Check JSON serialization
    json_str = report.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["benchmark_name"] == "ai_research_copilot_system_benchmark"
    assert "variant_a_bm25" in parsed["variants"]

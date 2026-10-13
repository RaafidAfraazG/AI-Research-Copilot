"""
Tests for evaluation benchmark datasets and ground-truth schemas (Phase 7).
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from evaluation.system_evaluator import BenchmarkQueryItem, SystemEvaluator


def test_retrieval_benchmark_json_integrity() -> None:
    path = Path(__file__).parent.parent.parent / "data" / "evaluation" / "retrieval_benchmark.json"
    assert path.exists(), f"Benchmark file missing at {path}"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["benchmark_name"] == "research_retrieval_benchmark_v1"
    assert data["total_queries"] >= 25
    queries = data["queries"]
    assert len(queries) >= 25

    intents_seen = set()
    for q in queries:
        assert q["query_id"]
        assert len(q["query"]) > 10
        assert q["intent"] in {
            "FACTUAL", "EXPLANATORY", "COMPARATIVE", "CAUSAL",
            "PROCEDURAL", "MULTI_HOP", "SURVEY", "TREND"
        }
        intents_seen.add(q["intent"])
        assert len(q["relevant_documents"]) > 0
        assert q["ground_truth_source"]

    # All 8 intents must be represented
    assert len(intents_seen) == 8


def test_answer_benchmark_json_integrity() -> None:
    path = Path(__file__).parent.parent.parent / "data" / "evaluation" / "answer_benchmark.json"
    assert path.exists(), f"Answer benchmark file missing at {path}"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["benchmark_name"] == "research_answer_benchmark_v1"
    assert data["total_queries"] >= 25
    queries = data["queries"]
    assert len(queries) >= 25

    intents_seen = set()
    for q in queries:
        assert q["query_id"]
        assert len(q["query"]) > 10
        assert q["intent"] in {
            "FACTUAL", "EXPLANATORY", "COMPARATIVE", "CAUSAL",
            "PROCEDURAL", "MULTI_HOP", "SURVEY", "TREND"
        }
        intents_seen.add(q["intent"])
        assert len(q["expected_dimensions"]) >= 2
        assert len(q["relevant_documents"]) > 0
        assert q["ground_truth_source"]

    assert len(intents_seen) == 8


def test_system_evaluator_load_dataset() -> None:
    evaluator = SystemEvaluator()
    items = evaluator.load_dataset()

    assert len(items) >= 25
    for it in items:
        assert isinstance(it, BenchmarkQueryItem)
        assert it.query_id
        assert it.query
        assert it.relevant_documents

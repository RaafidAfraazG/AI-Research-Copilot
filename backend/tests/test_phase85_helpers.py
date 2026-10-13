import json
import sys
from pathlib import Path
import pytest

# Ensure project root is on sys.path for scripts import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.phase85_smoke_test import (
    audit_environment,
    save_reports,
    run_experiment_a,
    run_experiment_b,
)
from ranking.cross_encoder_reranker import CrossEncoderReranker
from core.config import Settings


def test_audit_environment_structure():
    """Verify audit_environment returns all required keys without exposing secrets."""
    audit = audit_environment()
    assert "python_version" in audit
    assert "torch" in audit
    assert "sentence_transformers" in audit
    assert "gemini_sdk" in audit
    assert "config" in audit
    # Ensure api_key itself is never present
    assert "gemini_api_key" not in audit["config"]
    assert "gemini_api_key_configured" in audit["config"]
    assert isinstance(audit["config"]["gemini_api_key_configured"], bool)


@pytest.mark.asyncio
async def test_experiment_a_graceful_handling():
    """Verify Experiment A handles missing sentence-transformers gracefully."""
    env_audit = {
        "sentence_transformers": {"installed": False, "version": "NOT INSTALLED"},
    }
    res = await run_experiment_a(env_audit)
    assert res["status"] == "NOT VALIDATED"
    assert res["mode"] == "fallback"
    assert "not installed" in res["reason"]


@pytest.mark.asyncio
async def test_experiment_b_not_configured_handling():
    """Verify Experiment B handles unconfigured API key gracefully."""
    env_audit = {
        "config": {"gemini_api_key_configured": False},
    }
    res = await run_experiment_b(env_audit)
    assert res["status"] == "NOT CONFIGURED"
    assert res["api_key_configured"] is False
    assert "not set" in res["reason"].lower()


def test_cross_encoder_reranker_warmup_fallback(monkeypatch):
    """Verify warmup() method on CrossEncoderReranker returns False when model load fails."""
    st = Settings(reranker_fallback_mode="hybrid_score")
    reranker = CrossEncoderReranker(settings=st)
    # Simulate load failure
    monkeypatch.setattr(reranker, "_load_model", lambda: None)
    ok = reranker.warmup()
    assert ok is False


def test_cross_encoder_reranker_warmup_success():
    """Verify warmup() method on CrossEncoderReranker returns True when model loads."""
    st = Settings(reranker_fallback_mode="hybrid_score")
    reranker = CrossEncoderReranker(settings=st)
    ok = reranker.warmup()
    assert isinstance(ok, bool)


def test_save_reports(tmp_path, monkeypatch):
    """Verify save_reports generates valid JSON and Markdown files."""
    import scripts.phase85_smoke_test as s_mod

    monkeypatch.setattr(s_mod, "_PROJECT_ROOT", tmp_path)

    env_audit = audit_environment()
    exp_a = {
        "status": "NOT VALIDATED",
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "mode": "fallback",
        "device": "cpu",
        "cold_start_ms": 0.0,
        "warmup_succeeded": False,
        "warm_latency": {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "iterations": 0},
        "reason": "PyTorch not installed",
    }
    exp_b = {
        "status": "NOT CONFIGURED",
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "mode": "real",
        "api_key_configured": False,
        "smoke_request": {"status": "NOT RUN", "latency_ms": 0.0},
        "grounded_answer_test": {"status": "NOT RUN"},
        "insufficient_evidence_test": {"status": "NOT RUN", "refused_or_handled": False},
        "reason": "GEMINI_API_KEY not set",
    }

    json_p, md_p = save_reports(env_audit, exp_a, exp_b)

    assert json_p.exists()
    assert md_p.exists()

    with open(json_p, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["phase"] == "8.5"
        assert "truth_table" in data
        assert data["truth_table"]["offline_test_suite"] == "PASS"

    with open(md_p, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Phase 8.5 Real Model Validation Report" in content

"""
Unit tests for Phase 8: Google Gemini provider, paid-provider safety guard,
neural reranker status tracking, and factory integration.

These tests run fully offline using mocks — no real API calls are made.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import Settings
from llm.factory import get_llm_provider
from llm.gemini_provider import GeminiProvider, _strip_json_fences
from llm.mock_provider import MockLLMProvider
from llm.models import (
    LLMConfigurationError,
    LLMGenerationError,
    LLMResponse,
    LLMUsage,
)


# ── _strip_json_fences utility ────────────────────────────────────────────────


def test_strip_json_fences_plain_json():
    """JSON without fences passes through unchanged."""
    raw = '{"key": "value"}'
    assert _strip_json_fences(raw) == raw


def test_strip_json_fences_with_json_tag():
    """Strips ```json ... ``` fences."""
    raw = '```json\n{"key": "value"}\n```'
    result = _strip_json_fences(raw)
    assert result == '{"key": "value"}'


def test_strip_json_fences_bare_backticks():
    """Strips bare ``` ... ``` fences."""
    raw = '```\n{"key": "value"}\n```'
    result = _strip_json_fences(raw)
    assert result == '{"key": "value"}'


def test_strip_json_fences_multiline():
    """Strips fences from multiline JSON."""
    raw = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
    result = _strip_json_fences(raw)
    assert json.loads(result) == {"a": 1, "b": 2}


# ── GeminiProvider: configuration ────────────────────────────────────────────


def test_gemini_provider_name():
    st = Settings(gemini_api_key="test-key", gemini_model="gemini-1.5-flash")
    p = GeminiProvider(settings=st)
    assert p.provider_name == "gemini"


def test_gemini_model_name_from_settings():
    st = Settings(gemini_api_key="test-key", gemini_model="gemini-1.5-pro")
    p = GeminiProvider(settings=st)
    assert p.model_name == "gemini-1.5-pro"


def test_gemini_model_name_override():
    st = Settings(gemini_api_key="test-key", gemini_model="gemini-1.5-flash")
    p = GeminiProvider(api_key="test-key", model_name="gemini-2.0-flash-exp", settings=st)
    assert p.model_name == "gemini-2.0-flash-exp"


def test_gemini_missing_api_key_raises_on_generate():
    """GeminiProvider raises LLMConfigurationError if no API key is set."""
    st = Settings(gemini_api_key=None)
    p = GeminiProvider(settings=st)
    # _get_client should raise since api_key is None
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        p._get_client()


def test_gemini_missing_sdk_raises_config_error():
    """GeminiProvider raises LLMConfigurationError if google-generativeai is missing."""
    st = Settings(gemini_api_key="test-key")
    p = GeminiProvider(settings=st)

    with patch("builtins.__import__", side_effect=ImportError("No module named 'google'")):
        with pytest.raises((LLMConfigurationError, ImportError)):
            p._get_client()


# ── GeminiProvider: mocked generate ──────────────────────────────────────────


def _make_mock_response(text: str, tokens_in: int = 10, tokens_out: int = 20) -> MagicMock:
    """Build a mock Gemini GenerateContentResponse."""
    resp = MagicMock()
    resp.text = text
    # usage_metadata
    usage = MagicMock()
    usage.prompt_token_count = tokens_in
    usage.candidates_token_count = tokens_out
    usage.total_token_count = tokens_in + tokens_out
    resp.usage_metadata = usage
    # candidates
    candidate = MagicMock()
    candidate.finish_reason = "STOP"
    resp.candidates = [candidate]
    return resp


@pytest.mark.asyncio
async def test_gemini_generate_text_mode():
    """Generate in text mode returns parsed LLMResponse."""
    mock_resp = _make_mock_response("RAG combines retrieval and generation.")

    with (
        patch("llm.gemini_provider.GeminiProvider._get_client") as mock_client,
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
    ):
        mock_thread.return_value = mock_resp
        st = Settings(gemini_api_key="test-key", gemini_model="gemini-1.5-flash")
        p = GeminiProvider(settings=st)

        resp = await p.generate("What is RAG?", temperature=0.0)

    assert isinstance(resp, LLMResponse)
    assert resp.provider == "gemini"
    assert resp.model == "gemini-1.5-flash"
    assert "RAG" in resp.content
    assert resp.usage is not None
    assert resp.usage.total_tokens == 30
    assert resp.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_gemini_generate_json_mode_strips_fences():
    """JSON mode strips markdown fences from response."""
    json_content = '{"summary": "RAG is great.", "claims": []}'
    mock_resp = _make_mock_response(f"```json\n{json_content}\n```")

    with (
        patch("llm.gemini_provider.GeminiProvider._get_client"),
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
    ):
        mock_thread.return_value = mock_resp
        st = Settings(gemini_api_key="test-key")
        p = GeminiProvider(settings=st)

        resp = await p.generate("Summarize.", response_format="json")

    # Fences should be stripped and content should be valid JSON
    data = json.loads(resp.content)
    assert data["summary"] == "RAG is great."


@pytest.mark.asyncio
async def test_gemini_generate_with_system_prompt():
    """System prompt is passed to the model."""
    mock_resp = _make_mock_response("Sure!")

    captured_kwargs: dict = {}

    async def fake_to_thread(fn, *args, **kwargs):
        captured_kwargs.update({"called": True})
        return mock_resp

    with (
        patch("llm.gemini_provider.GeminiProvider._get_client"),
        patch("asyncio.to_thread", side_effect=fake_to_thread),
    ):
        st = Settings(gemini_api_key="test-key")
        p = GeminiProvider(settings=st)
        await p.generate(
            "Hello",
            system_prompt="You are a helpful assistant.",
            temperature=0.0,
        )

    assert captured_kwargs.get("called")


@pytest.mark.asyncio
async def test_gemini_generate_failure_raises_generation_error():
    """Errors during generation are wrapped in LLMGenerationError."""
    with (
        patch("llm.gemini_provider.GeminiProvider._get_client"),
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
    ):
        mock_thread.side_effect = RuntimeError("Network error")
        st = Settings(gemini_api_key="test-key")
        p = GeminiProvider(settings=st)

        with pytest.raises(LLMGenerationError, match="Gemini generation failed"):
            await p.generate("Hello")


# ── Factory: paid-provider safety guard ──────────────────────────────────────


def test_factory_blocks_openai_when_guard_active():
    """OpenAI is blocked when LLM_ALLOW_PAID_PROVIDER=false."""
    st = Settings(llm_allow_paid_provider=False)
    with pytest.raises(LLMConfigurationError, match="LLM_ALLOW_PAID_PROVIDER"):
        get_llm_provider("openai", settings=st)


def test_factory_blocks_anthropic_when_guard_active():
    """Anthropic is blocked when LLM_ALLOW_PAID_PROVIDER=false."""
    st = Settings(llm_allow_paid_provider=False)
    with pytest.raises(LLMConfigurationError, match="LLM_ALLOW_PAID_PROVIDER"):
        get_llm_provider("anthropic", settings=st)


def test_factory_allows_openai_when_guard_disabled():
    """OpenAI is allowed when LLM_ALLOW_PAID_PROVIDER=true."""
    st = Settings(llm_allow_paid_provider=True, openai_api_key=None)
    from llm.openai_provider import OpenAIProvider
    p = get_llm_provider("openai", settings=st)
    assert isinstance(p, OpenAIProvider)


def test_factory_gemini_always_allowed():
    """Gemini is always allowed regardless of guard setting."""
    st = Settings(llm_allow_paid_provider=False, gemini_api_key="test-key")
    p = get_llm_provider("gemini", settings=st)
    assert isinstance(p, GeminiProvider)


def test_factory_mock_always_allowed():
    """Mock is always allowed regardless of guard setting."""
    st = Settings(llm_allow_paid_provider=False)
    p = get_llm_provider("mock", settings=st)
    assert isinstance(p, MockLLMProvider)


def test_factory_gemini_as_default_provider():
    """Factory creates GeminiProvider when llm_provider=gemini."""
    st = Settings(llm_provider="gemini", gemini_api_key="test-key")
    p = get_llm_provider(settings=st)
    assert isinstance(p, GeminiProvider)
    assert p.provider_name == "gemini"


def test_factory_unknown_provider_raises():
    """Factory raises LLMConfigurationError for unknown provider names."""
    st = Settings(llm_provider="mock")
    with pytest.raises(LLMConfigurationError, match="Unknown LLM provider"):
        get_llm_provider("totally_unknown", settings=st)


# ── Config: new Gemini settings ───────────────────────────────────────────────


def test_config_gemini_defaults(monkeypatch):
    """Settings correctly expose Gemini defaults."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    st = Settings(gemini_api_key=None, gemini_model="gemini-flash-latest")
    assert st.gemini_model == "gemini-flash-latest"
    assert not st.gemini_api_key
    assert st.llm_allow_paid_provider is False


def test_config_gemini_from_env(monkeypatch):
    """Settings reads GEMINI_API_KEY from environment."""
    monkeypatch.setenv("GEMINI_API_KEY", "my-test-api-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")
    from core.config import get_settings
    get_settings.cache_clear()
    st = get_settings()
    assert st.gemini_api_key == "my-test-api-key"
    assert st.gemini_model == "gemini-1.5-pro"
    get_settings.cache_clear()


def test_config_llm_provider_includes_gemini():
    """llm_provider field accepts 'gemini' as a valid literal."""
    st = Settings(llm_provider="gemini")
    assert st.llm_provider == "gemini"


def test_config_llm_allow_paid_provider_env(monkeypatch):
    """LLM_ALLOW_PAID_PROVIDER env var is parsed correctly."""
    monkeypatch.setenv("LLM_ALLOW_PAID_PROVIDER", "true")
    from core.config import get_settings
    get_settings.cache_clear()
    st = get_settings()
    assert st.llm_allow_paid_provider is True
    get_settings.cache_clear()


# ── Reranker: fallback status is clearly tracked ──────────────────────────────


@pytest.mark.asyncio
async def test_reranker_fallback_flag_set_when_model_not_loaded():
    """When model load fails, reranker metadata clearly marks reranker_fallback=True."""
    from ranking.cross_encoder_reranker import CrossEncoderReranker
    from retrieval.models import ScoredChunk

    reranker = CrossEncoderReranker(model_name="nonexistent/model-does-not-exist",
                                    fallback_mode="hybrid_score")
    # Simulate model load failure
    reranker._load_failed = True
    reranker._load_error_msg = "No such model"

    candidates = [
        ScoredChunk(
            chunk_id="c1", document_id="d1", source_type="arxiv", source_id="x",
            content="Test content.", score=0.5, rank=1, retriever_name="hybrid",
        )
    ]
    reranked = await reranker.rerank("test query", candidates)
    assert len(reranked) == 1
    assert reranked[0].metadata.get("reranker_fallback") is True, (
        "Fallback flag must be set in metadata when cross-encoder model is not loaded"
    )


@pytest.mark.asyncio
async def test_reranker_no_fallback_flag_when_model_works():
    """When model returns real scores, reranker_fallback must NOT be in metadata."""
    from ranking.cross_encoder_reranker import CrossEncoderReranker
    from retrieval.models import ScoredChunk

    reranker = CrossEncoderReranker(model_name="test-model")
    reranker.score_pairs = MagicMock(return_value=[8.5, 1.2])  # type: ignore

    candidates = [
        ScoredChunk(
            chunk_id="c1", document_id="d1", source_type="arxiv", source_id="x",
            content="Dense passage retrieval uses dual encoders.", score=0.5, rank=1,
            retriever_name="hybrid",
        ),
        ScoredChunk(
            chunk_id="c2", document_id="d2", source_type="arxiv", source_id="y",
            content="Off-topic content about cooking.", score=0.3, rank=2,
            retriever_name="hybrid",
        ),
    ]
    reranked = await reranker.rerank("retrieval-augmented generation", candidates)
    # No fallback flag when real scores are returned
    assert not reranked[0].metadata.get("reranker_fallback"), (
        "reranker_fallback flag must NOT be set when cross-encoder returns real scores"
    )
    # c1 should win with score 8.5
    assert reranked[0].chunk_id == "c1"
    assert reranked[0].rerank_score == 8.5

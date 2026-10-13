"""
Unit tests for LLM provider abstraction, factory, and mock/concrete providers.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from core.config import Settings
from llm.base import LLMProvider
from llm.factory import get_llm_provider
from llm.mock_provider import MockLLMProvider
from llm.models import (
    LLMConfigurationError,
    LLMGenerationError,
    LLMResponse,
)
from llm.ollama_provider import OllamaProvider
from llm.openai_provider import OpenAIProvider


@pytest.mark.asyncio
async def test_mock_llm_provider_generation():
    """Verify MockLLMProvider produces deterministic structured output."""
    provider = MockLLMProvider()
    prompt = """USER QUESTION: What is RAG?
RETRIEVED EVIDENCE PASSAGES:
[EVIDENCE_ID: evidence_001]
Title: RAG Paper
Passage: Retrieval-augmented generation combines parametric and non-parametric memory.
[EVIDENCE_ID: evidence_002]
Title: DPR Paper
Passage: Dense passage retrieval uses dual-encoders for open-domain QA.
"""
    resp = await provider.generate(prompt=prompt, response_format="json")

    assert resp.provider == "mock"
    assert resp.latency_ms >= 0.0
    data = json.loads(resp.content)
    assert "summary" in data
    assert len(data["claims"]) >= 2
    assert data["claims"][0]["evidence_ids"] == ["evidence_001"]
    assert data["claims"][1]["evidence_ids"] == ["evidence_002"]


@pytest.mark.asyncio
async def test_mock_llm_provider_no_evidence():
    """Verify MockLLMProvider sets insufficient_evidence when prompt has no evidence IDs."""
    provider = MockLLMProvider()
    resp = await provider.generate("USER QUESTION: Something with no evidence", response_format="json")
    data = json.loads(resp.content)
    assert data["insufficient_evidence"] is True
    assert "Insufficient evidence" in data["summary"]


def test_llm_factory_selection():
    """Verify factory returns correct provider instances."""
    st = Settings(llm_provider="mock")
    mock_p = get_llm_provider(settings=st)
    assert isinstance(mock_p, MockLLMProvider)

    ollama_p = get_llm_provider("ollama", settings=st)
    assert isinstance(ollama_p, OllamaProvider)

    with pytest.raises(LLMConfigurationError, match="Unknown LLM provider"):
        get_llm_provider("unsupported_provider", settings=st)


@pytest.mark.asyncio
async def test_openai_provider_missing_key_raises_error():
    """Verify OpenAIProvider raises LLMConfigurationError if API key is missing."""
    st = Settings(openai_api_key=None)
    provider = OpenAIProvider(settings=st)

    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY is not configured"):
        await provider.generate("Test prompt")


@pytest.mark.asyncio
async def test_openai_provider_success():
    """Verify OpenAIProvider parses chat completion response."""
    st = Settings(openai_api_key="test-sk-key", openai_model="gpt-4o-mini")
    provider = OpenAIProvider(settings=st)

    mock_resp = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": '{"summary": "Test response"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await provider.generate("Test prompt", response_format="json")

        assert res.provider == "openai"
        assert res.model == "gpt-4o-mini"
        assert res.content == '{"summary": "Test response"}'
        assert res.usage.total_tokens == 15

"""
Factory for instantiating LLM providers based on environment configuration.

Free-tier safety:
    LLM_ALLOW_PAID_PROVIDER=false (default) prevents accidentally using
    paid providers (OpenAI, Anthropic). Gemini free-tier and Mock are
    always permitted.
"""
from __future__ import annotations

from typing import Literal
import structlog

from core.config import Settings, get_settings
from llm.anthropic_provider import AnthropicProvider
from llm.base import LLMProvider
from llm.gemini_provider import GeminiProvider
from llm.mock_provider import MockLLMProvider
from llm.models import LLMConfigurationError
from llm.ollama_provider import OllamaProvider
from llm.openai_provider import OpenAIProvider

logger = structlog.get_logger(__name__)

# Providers that have paid-only API tiers — blocked unless LLM_ALLOW_PAID_PROVIDER=true
_PAID_PROVIDERS: frozenset[str] = frozenset({"openai", "anthropic"})

# Ollama is local / self-hosted — allowed in both modes
_LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama"})


def get_llm_provider(
    provider_name: Literal["ollama", "openai", "anthropic", "gemini", "mock"] | str | None = None,
    settings: Settings | None = None,
) -> LLMProvider:
    """
    Instantiate and return the configured LLMProvider.

    Args:
        provider_name: Optional override.
                       Accepted values: 'gemini', 'ollama', 'openai', 'anthropic', 'mock'.
        settings:      Optional Settings instance.

    Returns:
        LLMProvider instance.

    Raises:
        LLMConfigurationError: If the requested provider is unknown, or if a paid
                                provider is requested while LLM_ALLOW_PAID_PROVIDER=false.
    """
    st = settings or get_settings()
    p_name = (provider_name or st.llm_provider).lower().strip()

    # ── Paid-provider safety guard ────────────────────────────────────────────
    if p_name in _PAID_PROVIDERS and not st.llm_allow_paid_provider:
        raise LLMConfigurationError(
            f"Provider '{p_name}' requires a paid API key and LLM_ALLOW_PAID_PROVIDER is false. "
            "To use this provider set LLM_ALLOW_PAID_PROVIDER=true in .env. "
            "Use LLM_PROVIDER=gemini (free tier) or LLM_PROVIDER=mock for offline testing."
        )

    # ── Provider instantiation ────────────────────────────────────────────────
    if p_name == "gemini":
        return GeminiProvider(settings=st)
    elif p_name == "ollama":
        return OllamaProvider(settings=st)
    elif p_name == "openai":
        return OpenAIProvider(settings=st)
    elif p_name == "anthropic":
        return AnthropicProvider(settings=st)
    elif p_name == "mock":
        return MockLLMProvider()
    else:
        raise LLMConfigurationError(
            f"Unknown LLM provider '{p_name}'. "
            "Supported providers: 'gemini', 'ollama', 'openai', 'anthropic', 'mock'."
        )

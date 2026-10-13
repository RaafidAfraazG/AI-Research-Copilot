"""
Abstract base class for LLM providers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from llm.models import LLMRequest, LLMResponse


class LLMProvider(ABC):
    """
    Abstract interface for provider-agnostic LLM generation.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique provider name (e.g. 'ollama', 'openai', 'anthropic', 'mock')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: User/task prompt content.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Max tokens to generate.
            response_format: 'text' or 'json'.
            
        Returns:
            Standardized LLMResponse.
        """
        pass

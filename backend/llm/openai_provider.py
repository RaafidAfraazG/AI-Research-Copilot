"""
OpenAI LLM provider implementation.
Supports GPT-4o, GPT-4o-mini, and compatible chat completion endpoints.
"""
from __future__ import annotations

import time
from typing import Any, Literal
import httpx
import structlog

from core.config import Settings, get_settings
from llm.base import LLMProvider
from llm.models import (
    LLMConfigurationError,
    LLMGenerationError,
    LLMResponse,
    LLMUsage,
)

logger = structlog.get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    Provider for OpenAI models using official SDK or HTTP client.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float = 60.0,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.api_key = api_key or self.settings.openai_api_key
        self._model_name = model_name or self.settings.openai_model
        self.timeout = timeout_seconds

        if not self.api_key:
            logger.warning("openai_api_key_not_configured")

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> LLMResponse:
        """Execute chat completion against OpenAI API."""
        if not self.api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is not configured. Set OPENAI_API_KEY in .env or choose another provider."
            )

        start_time = time.perf_counter()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            raw_usage = data.get("usage", {})
            usage = LLMUsage(
                input_tokens=raw_usage.get("prompt_tokens"),
                output_tokens=raw_usage.get("completion_tokens"),
                total_tokens=raw_usage.get("total_tokens"),
            )

            logger.info(
                "openai_generation_completed",
                model=self._model_name,
                latency_ms=latency_ms,
                total_tokens=usage.total_tokens,
            )

            return LLMResponse(
                content=content,
                provider="openai",
                model=self._model_name,
                latency_ms=latency_ms,
                usage=usage,
                finish_reason=finish_reason,
            )

        except Exception as e:
            logger.error("openai_generation_failed", error=str(e), model=self._model_name)
            raise LLMGenerationError(f"OpenAI generation failed: {e}") from e

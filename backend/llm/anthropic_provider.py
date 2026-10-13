"""
Anthropic LLM provider implementation.
Supports Claude-3.5-Sonnet, Claude-3-Haiku, and Messages API.
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


class AnthropicProvider(LLMProvider):
    """
    Provider for Anthropic Claude models using the Messages API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float = 60.0,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.api_key = api_key or self.settings.anthropic_api_key
        self._model_name = model_name or self.settings.anthropic_model
        self.timeout = timeout_seconds

        if not self.api_key:
            logger.warning("anthropic_api_key_not_configured")

    @property
    def provider_name(self) -> str:
        return "anthropic"

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
        """Execute Messages API completion against Anthropic."""
        if not self.api_key:
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY is not configured. Set ANTHROPIC_API_KEY in .env or choose another provider."
            )

        start_time = time.perf_counter()
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": max_tokens or 2048,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            # Extract text blocks
            content_blocks = data.get("content", [])
            text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
            content = "".join(text_parts)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            raw_usage = data.get("usage", {})
            input_tokens = raw_usage.get("input_tokens")
            output_tokens = raw_usage.get("output_tokens")
            total_tokens = (input_tokens + output_tokens) if (input_tokens and output_tokens) else None

            usage = LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

            logger.info(
                "anthropic_generation_completed",
                model=self._model_name,
                latency_ms=latency_ms,
                total_tokens=total_tokens,
            )

            return LLMResponse(
                content=content,
                provider="anthropic",
                model=self._model_name,
                latency_ms=latency_ms,
                usage=usage,
                finish_reason=data.get("stop_reason"),
            )

        except Exception as e:
            logger.error("anthropic_generation_failed", error=str(e), model=self._model_name)
            raise LLMGenerationError(f"Anthropic generation failed: {e}") from e

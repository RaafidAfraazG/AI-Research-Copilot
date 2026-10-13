"""
Ollama local LLM provider implementation.
Communicates asynchronously with a local Ollama server via HTTP REST API.
"""
from __future__ import annotations

import time
from typing import Any, Literal
import httpx
import structlog

from core.config import Settings, get_settings
from llm.base import LLMProvider
from llm.models import LLMGenerationError, LLMResponse, LLMUsage

logger = structlog.get_logger(__name__)


class OllamaProvider(LLMProvider):
    """
    Provider for locally hosted models via Ollama (e.g. llama3.2, mistral, qwen2.5).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float = 60.0,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = (base_url or self.settings.ollama_base_url).rstrip("/")
        self._model_name = model_name or self.settings.ollama_model
        self.timeout = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "ollama"

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
        """Execute completion on local Ollama server."""
        start_time = time.perf_counter()
        endpoint = f"{self.base_url}/api/generate"

        payload: dict[str, Any] = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if response_format == "json":
            payload["format"] = "json"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()

            content = data.get("response", "")
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Extract Ollama token metrics
            prompt_tokens = data.get("prompt_eval_count")
            output_tokens = data.get("eval_count")
            total_tokens = (prompt_tokens + output_tokens) if (prompt_tokens and output_tokens) else None

            usage = LLMUsage(
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

            logger.info(
                "ollama_generation_completed",
                model=self._model_name,
                latency_ms=latency_ms,
                total_tokens=total_tokens,
            )

            return LLMResponse(
                content=content,
                provider="ollama",
                model=self._model_name,
                latency_ms=latency_ms,
                usage=usage,
                finish_reason="stop" if data.get("done") else None,
                raw_response={"total_duration": data.get("total_duration")},
            )

        except httpx.ConnectError as e:
            raise LLMGenerationError(
                f"Could not connect to Ollama at {self.base_url}. Ensure Ollama is running (`ollama serve`)."
            ) from e
        except Exception as e:
            logger.error("ollama_generation_failed", error=str(e), model=self._model_name)
            raise LLMGenerationError(f"Ollama generation failed: {e}") from e

"""
Google Gemini LLM provider — free-tier primary provider for Phase 8.

Supported models (free tier):
  - gemini-1.5-flash          (primary: fast, generous free quota)
  - gemini-1.5-flash-8b       (smaller variant)
  - gemini-1.5-pro            (higher quality, lower free quota)
  - gemini-2.0-flash-exp      (experimental)

Configuration:
  LLM_PROVIDER=gemini
  GEMINI_API_KEY=<your-key-from-aistudio.google.com>
  GEMINI_MODEL=gemini-1.5-flash

Safety:
  LLM_ALLOW_PAID_PROVIDER=false  (default) — prevents accidental paid API use.
  The Gemini free tier is explicitly allowed regardless of this setting.
"""
from __future__ import annotations

import json
import time
from typing import Any, Literal

import structlog

from llm.base import LLMProvider
from llm.models import (
    LLMConfigurationError,
    LLMGenerationError,
    LLMResponse,
    LLMUsage,
)

logger = structlog.get_logger(__name__)

# Free-tier Gemini model identifiers (not subject to paid-provider guard)
_FREE_TIER_GEMINI_MODELS: frozenset[str] = frozenset(
    {
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash-002",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-lite",
        "gemini-exp-1206",
    }
)


class GeminiProvider(LLMProvider):
    """
    LLM provider backed by Google Gemini (google-generativeai SDK).

    Free-tier compliant: uses gemini-1.5-flash by default which has a
    generous daily quota with no cost.

    Args:
        api_key:        Gemini API key (GEMINI_API_KEY env var).
        model_name:     Gemini model identifier.
        timeout_seconds: HTTP timeout.
        settings:       Optional Settings override.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float = 60.0,
        settings: Any = None,
    ) -> None:
        from core.config import get_settings

        self.settings = settings or get_settings()
        self.api_key = api_key or self.settings.gemini_api_key
        self._model_name = model_name or self.settings.gemini_model
        self.timeout = timeout_seconds

        if not self.api_key:
            logger.warning(
                "gemini_api_key_not_configured",
                hint="Set GEMINI_API_KEY in .env — get a free key at https://aistudio.google.com",
            )

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Lazily import and configure the Gemini SDK client."""
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise LLMConfigurationError(
                "google-generativeai is not installed. Run: pip install google-generativeai"
            ) from e

        # Treat both None and empty string as "not configured"
        if not self.api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not configured. "
                "Get a free key at https://aistudio.google.com and set it in .env."
            )

        genai.configure(api_key=self.api_key)
        return genai

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: str | None,
    ) -> tuple[str | None, str]:
        """
        Gemini SDK uses system_instruction separately from the user prompt.
        Returns (system_instruction, user_content).
        """
        return system_prompt, prompt

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Extract text from a Gemini SDK GenerateContentResponse."""
        try:
            return response.text
        except AttributeError:
            # Fallback: iterate candidates
            for candidate in getattr(response, "candidates", []):
                for part in getattr(candidate.content, "parts", []):
                    if hasattr(part, "text"):
                        return part.text
        return ""

    @staticmethod
    def _extract_usage(response: Any) -> LLMUsage | None:
        """Extract token usage from response metadata."""
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is None:
            return None
        return LLMUsage(
            input_tokens=getattr(usage_meta, "prompt_token_count", None),
            output_tokens=getattr(usage_meta, "candidates_token_count", None),
            total_tokens=getattr(usage_meta, "total_token_count", None),
        )

    # ── Public API ────────────────────────────────────────────────────────────

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
        Generate a completion using Google Gemini.

        Notes:
        - Runs synchronous SDK call in asyncio.to_thread to stay non-blocking.
        - JSON mode is approximated by injecting an instruction into system_prompt
          (Gemini does not have an explicit JSON-mode flag in the base SDK).
        """
        import asyncio

        genai = self._get_client()
        sys_instr, user_content = self._build_messages(prompt, system_prompt)

        # Inject JSON instruction when caller requests structured output
        if response_format == "json":
            json_hint = (
                "You must respond with valid JSON only. "
                "Do not include markdown fences, explanations, or any text outside the JSON object."
            )
            sys_instr = f"{sys_instr}\n\n{json_hint}" if sys_instr else json_hint

        generation_config: dict[str, Any] = {
            "temperature": temperature,
        }
        if response_format == "json":
            generation_config["response_mime_type"] = "application/json"
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        def _sync_generate() -> Any:
            model_kwargs: dict[str, Any] = {
                "model_name": self._model_name,
                "generation_config": generation_config,
            }
            if sys_instr:
                model_kwargs["system_instruction"] = sys_instr

            model = genai.GenerativeModel(**model_kwargs)
            return model.generate_content(user_content)

        start_time = time.perf_counter()
        try:
            response = await asyncio.to_thread(_sync_generate)
        except Exception as e:
            logger.error(
                "gemini_generation_failed",
                model=self._model_name,
                error=str(e),
            )
            raise LLMGenerationError(f"Gemini generation failed: {e}") from e

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        content = self._extract_content(response)
        usage = self._extract_usage(response)

        # Strip common markdown fences that Gemini sometimes adds despite instruction
        if response_format == "json":
            content = _strip_json_fences(content)

        # Validate JSON when requested (fail fast with clear error)
        if response_format == "json":
            try:
                json.loads(content)
            except json.JSONDecodeError as jde:
                logger.warning(
                    "gemini_json_parse_failed",
                    model=self._model_name,
                    content_snippet=content[:200],
                    error=str(jde),
                )
                # Return raw; caller can decide to retry
                # Don't raise — partial content may still be usable

        finish_reason = None
        for candidate in getattr(response, "candidates", []):
            finish_reason_raw = getattr(candidate, "finish_reason", None)
            if finish_reason_raw is not None:
                finish_reason = str(finish_reason_raw)
            break

        logger.info(
            "gemini_generation_completed",
            model=self._model_name,
            latency_ms=latency_ms,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
        )

        return LLMResponse(
            content=content,
            provider="gemini",
            model=self._model_name,
            latency_ms=latency_ms,
            usage=usage,
            finish_reason=finish_reason,
        )


# ── Utilities ─────────────────────────────────────────────────────────────────


def _strip_json_fences(text: str) -> str:
    """
    Remove markdown code fences (```json ... ``` or ``` ... ```) from text.
    Gemini occasionally wraps JSON in fences despite being told not to.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence (with optional language tag)
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped

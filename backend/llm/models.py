"""
Data models and exceptions for LLM providers.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class LLMError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when an LLM provider is improperly configured (e.g. missing API key)."""
    pass


class LLMGenerationError(LLMError):
    """Raised when generation fails at runtime."""
    pass


class LLMUsage(BaseModel):
    """Token usage metrics."""
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class LLMRequest(BaseModel):
    """Standardized request payload for LLM providers."""
    prompt: str = Field(..., description="User prompt content")
    system_prompt: str | None = Field(default=None, description="Optional system instructions")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(default=2048, ge=1, le=8192, description="Maximum tokens to generate")
    response_format: Literal["text", "json"] = Field(default="text", description="Expected output format")


class LLMResponse(BaseModel):
    """Standardized response payload from LLM providers."""
    content: str = Field(..., description="Generated text content")
    provider: str = Field(..., description="Provider name (e.g. 'ollama', 'openai', 'anthropic', 'mock')")
    model: str = Field(..., description="Model identifier used for generation")
    latency_ms: float = Field(default=0.0, description="Inference latency in milliseconds")
    usage: LLMUsage | None = Field(default=None, description="Token usage statistics if available")
    finish_reason: str | None = Field(default=None, description="Generation stop reason")
    raw_response: dict[str, Any] | None = Field(default=None, description="Raw provider response metadata")

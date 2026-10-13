"""
OpenAI embedding provider using the official OpenAI API or httpx.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


class OpenAIEmbeddingProvider:
    """
    Embedding provider utilizing OpenAI's embeddings API (e.g. text-embedding-3-small).
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._dimension = dimension
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self.aembed_texts(texts))

    def embed_query(self, text: str) -> list[float]:
        return asyncio.run(self.aembed_query(text))

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "input": texts,
            "model": self._model_name,
        }
        if "text-embedding-3" in self._model_name and self._dimension:
            payload["dimensions"] = self._dimension

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            # Sort by index to maintain ordering
            sorted_items = sorted(data["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in sorted_items]

    async def aembed_query(self, text: str) -> list[float]:
        res = await self.aembed_texts([text])
        return res[0] if res else [0.0] * self._dimension

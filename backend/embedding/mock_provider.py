"""
Mock embedding provider for unit testing and offline development.
Generates deterministic normalized pseudo-embeddings based on text hashing.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence


class MockEmbeddingProvider:
    """
    Deterministic embedding provider for testing.
    Computes a normalized pseudo-random vector from MD5 hash seeds.
    """

    def __init__(self, dimension: int = 384, model_name: str = "mock-minilm-384") -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _generate_vector(self, text: str) -> list[float]:
        if not text:
            # Return zero-like normalized vector
            return [0.0] * self._dimension

        # Generate deterministic float values from text hash
        raw: list[float] = []
        for i in range(self._dimension):
            seed = f"{text}:{i}"
            h = hashlib.md5(seed.encode("utf-8")).hexdigest()
            # Convert first 8 hex chars to signed float in [-1.0, 1.0]
            val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
            raw.append(val)

        # L2-normalize
        norm = math.sqrt(sum(x * x for x in raw))
        if norm > 0:
            return [x / norm for x in raw]
        return raw

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._generate_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._generate_vector(text)

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

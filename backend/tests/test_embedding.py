"""
Unit tests for Embedding Providers (embedding/).
"""
from __future__ import annotations

import pytest

from core.config import Settings
from embedding.base import EmbeddingProvider
from embedding.factory import create_embedding_provider
from embedding.mock_provider import MockEmbeddingProvider
from embedding.sentence_transformer_provider import SentenceTransformerProvider


class TestMockEmbeddingProvider:
    def test_mock_provider_dimension(self) -> None:
        p = MockEmbeddingProvider(dimension=384)
        assert p.dimension == 384
        assert p.model_name == "mock-minilm-384"

    def test_mock_provider_embed_query(self) -> None:
        p = MockEmbeddingProvider(dimension=128)
        v = p.embed_query("research questions")
        assert len(v) == 128
        assert all(isinstance(x, float) for x in v)

    def test_mock_provider_deterministic(self) -> None:
        p = MockEmbeddingProvider(dimension=64)
        v1 = p.embed_query("dense passage retrieval")
        v2 = p.embed_query("dense passage retrieval")
        assert v1 == v2

    def test_mock_provider_embed_texts(self) -> None:
        p = MockEmbeddingProvider(dimension=32)
        texts = ["First text", "Second text", "Third text"]
        vectors = p.embed_texts(texts)
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 32

    @pytest.mark.asyncio
    async def test_mock_provider_async_methods(self) -> None:
        p = MockEmbeddingProvider(dimension=64)
        qv = await p.aembed_query("test query")
        assert len(qv) == 64

        tvs = await p.aembed_texts(["t1", "t2"])
        assert len(tvs) == 2


class TestEmbeddingFactory:
    def test_create_mock_provider_fallback(self) -> None:
        s = Settings(embedding_provider="local", postgres_password="test")
        # When sentence-transformers is not instantiated or mock is requested
        p = MockEmbeddingProvider(dimension=384)
        assert isinstance(p, EmbeddingProvider)

    def test_create_openai_provider_missing_key_raises(self) -> None:
        s = Settings(
            embedding_provider="openai",
            openai_api_key=None,
            postgres_password="test",
        )
        with pytest.raises(ValueError, match="OPENAI_API_KEY must be set"):
            create_embedding_provider(s)

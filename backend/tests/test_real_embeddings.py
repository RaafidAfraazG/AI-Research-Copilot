"""
Opt-in real-model integration tests (RUN_REAL_MODEL_TESTS=1).

They load the real all-MiniLM-L6-v2 and cross-encoder weights from the local
HuggingFace cache (or download them) and therefore are skipped by default.
"""
from __future__ import annotations

import pytest

from core.config import get_settings
from embedding.identity import describe_embedding_provider
from embedding.sentence_transformer_provider import SentenceTransformerProvider
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.index_admin import rebuild_index
from retrieval.index_manifest import EmbeddingModelMismatchError, IndexStore
from retrieval.models import RetrievalQuery
from embedding.mock_provider import MockEmbeddingProvider

from tests.conftest import MOCK_CORPUS

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="module")
def real_provider():
    return SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")


def test_real_provider_identity(real_provider):
    ident = describe_embedding_provider(real_provider)
    assert ident.model_name == "all-MiniLM-L6-v2"
    assert ident.dimension == 384 and ident.is_mock is False and ident.normalized


@pytest.fixture
def real_index(tmp_path, real_provider, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    get_settings.cache_clear()
    store = IndexStore(settings=get_settings())
    rep = rebuild_index(store, real_provider, MOCK_CORPUS, settings=get_settings())
    assert rep["status"] == "complete", rep
    return store


@pytest.mark.asyncio
async def test_dense_uses_real_vectors_and_ranks_semantically(real_index, real_provider):
    dense = DenseRetriever(embedding_provider=real_provider, index_store=real_index)
    dense._server_available = False
    res = await dense.search(RetrievalQuery(query_text="dual encoder for open-domain question answering", top_k=3))
    assert res.metadata["embedding_is_mock"] is False
    assert res.chunks[0].source_id == "2004.04906"
    assert 0.0 < res.chunks[0].dense_score <= 1.0


@pytest.mark.asyncio
async def test_hybrid_with_real_embeddings(real_index, real_provider):
    dense = DenseRetriever(embedding_provider=real_provider, index_store=real_index)
    dense._server_available = False
    bm25 = BM25Retriever(index_store=real_index)
    bm25._server_available = False
    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
    res = await hybrid.search(RetrievalQuery(query_text="self-attention without recurrence", top_k=3))
    assert res.chunks[0].source_id == "1706.03762"
    assert res.metadata["embedding_model"] == "all-MiniLM-L6-v2"


def test_mock_provider_rejected_by_real_index(real_index):
    with pytest.raises(EmbeddingModelMismatchError):
        real_index.validate_for_read(MockEmbeddingProvider(dimension=384))

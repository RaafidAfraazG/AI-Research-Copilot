"""
Integration tests for the retrieval pipeline against an isolated local index.

These run offline: the index is built in a per-test temporary data root with the
EXPLICITLY configured mock embedder (see conftest.mock_index). They verify wiring
(manifest -> BM25/dense/hybrid), not embedding quality. Real-model quality checks
live in tests/test_real_embeddings.py (opt-in, marker `real_model`).
"""
from __future__ import annotations

import pytest

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.indexing_service import IndexingService
from retrieval.models import IngestDocumentRequest, RetrievalQuery

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("mock_index")]


class TestRetrievalIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_indexing_and_bm25_search(self) -> None:
        service = IndexingService()
        req = IngestDocumentRequest(
            source_type="arxiv",
            source_id="test.integration.001",
            title="Transformer Scaling Laws for Reasoning Models",
            content="Scaling transformer parameters and training compute leads to predictable empirical performance improvements across diverse language tasks.",
            metadata={"category": "cs.LG", "year": 2024},
        )
        ingest_res = await service.index_document(req)
        assert ingest_res.status in {"success", "partial"}
        assert ingest_res.total_chunks >= 1

        bm25 = BM25Retriever()
        query = RetrievalQuery(query_text="transformer scaling compute parameters", top_k=3)
        res = await bm25.search(query)
        assert len(res.chunks) >= 1
        assert any("Transformer Scaling" in c.title for c in res.chunks)
        assert res.chunks[0].score > 0

    @pytest.mark.asyncio
    async def test_dense_vector_search_uses_stored_vectors(self) -> None:
        dense = DenseRetriever()
        query = RetrievalQuery(query_text="attention mechanisms and neural representations", top_k=3)
        res = await dense.search(query)
        assert res.retriever == "dense"
        assert len(res.chunks) >= 1
        assert res.chunks[0].dense_score is not None
        assert res.metadata["source"] == "local_qdrant_storage"
        assert res.metadata["embedding_is_mock"] is True  # honestly labeled

    @pytest.mark.asyncio
    async def test_hybrid_rrf_fusion_integration(self) -> None:
        hybrid = HybridRetriever()
        query = RetrievalQuery(query_text="retrieval augmented generation memory", top_k=5, alpha=0.5)
        res = await hybrid.search(query, fusion_method="rrf")
        assert res.retriever == "hybrid"
        assert len(res.chunks) >= 1
        top_chunk = res.chunks[0]
        assert top_chunk.score > 0
        assert top_chunk.retriever_name == "hybrid"
        assert top_chunk.rank == 1
        assert "fusion_method" in res.metadata
        assert res.metadata["index_generation"]

    @pytest.mark.asyncio
    async def test_hybrid_relative_score_fusion_integration(self) -> None:
        hybrid = HybridRetriever()
        query = RetrievalQuery(query_text="dense passage retrieval dual encoder", top_k=5, alpha=0.6)
        res = await hybrid.search(query, fusion_method="rsf")
        assert res.retriever == "hybrid"
        assert len(res.chunks) >= 1
        assert res.metadata.get("fusion_method") == "rsf"

"""
Unit tests for IndexingService.
"""
from __future__ import annotations

from unittest.mock import AsyncMock
import pytest

from embedding.mock_provider import MockEmbeddingProvider
from retrieval.indexing_service import IndexingService
from retrieval.models import IngestDocumentRequest


@pytest.mark.asyncio
class TestIndexingService:
    async def test_index_empty_document(self) -> None:
        service = IndexingService(
            embedding_provider=MockEmbeddingProvider(dimension=64),
        )
        req = IngestDocumentRequest(
            source_type="local",
            source_id="empty-1",
            title="Empty Doc",
            content="   ",
        )
        res = await service.index_document(req)
        assert res.total_chunks == 0
        assert res.status == "empty"

    async def test_index_document_pipeline_success(self) -> None:
        mock_os = AsyncMock()
        mock_os.bulk.return_value = {"errors": False, "items": []}

        mock_qd = AsyncMock()
        mock_qd.upsert.return_value = None

        provider = MockEmbeddingProvider(dimension=64)
        service = IndexingService(
            opensearch_client=mock_os,
            qdrant_client=mock_qd,
            embedding_provider=provider,
        )

        req = IngestDocumentRequest(
            source_type="arxiv",
            source_id="2005.11401",
            title="Retrieval-Augmented Generation",
            content="RAG combines parametric and non-parametric memory. It achieves state-of-the-art results.",
        )
        res = await service.index_document(req)
        assert res.status == "success"
        assert res.total_chunks >= 1
        assert res.indexed_in_opensearch is True
        assert res.indexed_in_qdrant is True
        assert mock_os.bulk.called
        assert mock_qd.upsert.called

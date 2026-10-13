"""
Unit tests for BM25 and Dense Retrievers.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from core.config import Settings
from embedding.mock_provider import MockEmbeddingProvider
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.models import RetrievalQuery


class TestBM25Retriever:
    def test_build_query_dsl_structure(self) -> None:
        settings = Settings(postgres_password="test", opensearch_index_name="docs_test")
        retriever = BM25Retriever(settings=settings)

        query = RetrievalQuery(
            query_text="RAG architecture vector database",
            top_k=15,
            source_types=["arxiv", "web"],
            filters={"category": "cs.CL"},
        )
        dsl = retriever.build_query(query)

        assert dsl["size"] == 15
        assert "_source" in dsl
        bool_q = dsl["query"]["bool"]
        assert len(bool_q["must"]) == 1
        multi_match = bool_q["must"][0]["multi_match"]
        assert multi_match["query"] == "RAG architecture vector database"
        assert "title^2.5" in multi_match["fields"]
        assert "content^1.0" in multi_match["fields"]

        filter_clauses = bool_q["filter"]
        assert len(filter_clauses) == 2
        assert {"terms": {"source_type": ["arxiv", "web"]}} in filter_clauses
        assert {"term": {"metadata.category": "cs.CL"}} in filter_clauses

    @pytest.mark.asyncio
    async def test_bm25_search_parses_hits_correctly(self) -> None:
        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "chunk-123",
                        "_score": 4.82,
                        "_source": {
                            "chunk_id": "chunk-123",
                            "document_id": "doc-abc",
                            "source_type": "arxiv",
                            "source_id": "2005.11401",
                            "title": "Retrieval-Augmented Generation",
                            "content": "RAG pairs seq2seq with dense index.",
                            "chunk_index": 0,
                            "total_chunks": 1,
                            "metadata": {"year": 2020},
                        },
                    }
                ],
            }
        }
        retriever = BM25Retriever(client=mock_client)
        result = await retriever.search(RetrievalQuery(query_text="RAG", top_k=5))

        assert result.retriever == "bm25"
        assert result.total_found == 1
        assert len(result.chunks) == 1
        c = result.chunks[0]
        assert c.chunk_id == "chunk-123"
        assert c.document_id == "doc-abc"
        assert c.score == 4.82
        assert c.bm25_score == 4.82
        assert c.bm25_rank == 1
        assert c.title == "Retrieval-Augmented Generation"


class TestDenseRetriever:
    def test_build_qdrant_filter(self) -> None:
        retriever = DenseRetriever(
            embedding_provider=MockEmbeddingProvider(dimension=64),
        )
        query = RetrievalQuery(
            query_text="Transformer self attention",
            source_types=["arxiv"],
            filters={"year": 2017},
        )
        qfilter = retriever._build_filter(query)
        assert qfilter is not None
        assert len(qfilter.must) == 2

    @pytest.mark.asyncio
    async def test_dense_search_parses_points_correctly(self, mock_index) -> None:
        mock_client = AsyncMock()
        mock_pt = MagicMock()
        mock_pt.id = "point-uuid-1"
        mock_pt.score = 0.892
        mock_pt.payload = {
            "index_generation": mock_index.active_generation_id(),
            "chunk_id": "point-uuid-1",
            "document_id": "doc-trans",
            "source_type": "arxiv",
            "source_id": "1706.03762",
            "title": "Attention Is All You Need",
            "content": "Self-attention mechanism without recurrence.",
            "chunk_index": 0,
            "total_chunks": 1,
            "metadata": {"year": 2017},
        }

        mock_search_result = MagicMock()
        mock_search_result.points = [mock_pt]
        mock_client.query_points.return_value = mock_search_result

        provider = MockEmbeddingProvider(dimension=384)
        retriever = DenseRetriever(
            client=mock_client, embedding_provider=provider, index_store=mock_index
        )
        result = await retriever.search(RetrievalQuery(query_text="Self attention", top_k=5))

        assert result.retriever == "dense"
        assert len(result.chunks) == 1
        c = result.chunks[0]
        assert c.chunk_id == "point-uuid-1"
        assert c.document_id == "doc-trans"
        assert c.score == 0.892
        assert c.dense_score == 0.892
        assert c.dense_rank == 1

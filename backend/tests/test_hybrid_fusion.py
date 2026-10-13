"""
Unit tests for Hybrid Fusion (Reciprocal Rank Fusion and Relative Score Fusion).
"""
from __future__ import annotations

from unittest.mock import AsyncMock
import pytest

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievalQuery, RetrievalResult, ScoredChunk


def make_chunk(
    chunk_id: str,
    doc_id: str,
    score: float,
    retriever_name: str = "bm25",
    rank: int = 1,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        source_type="arxiv",
        source_id="2024.0001",
        title=f"Paper {doc_id}",
        content=f"Content for {chunk_id}",
        score=score,
        retriever_name=retriever_name,
        rank=rank,
    )


class TestReciprocalRankFusion:
    def test_rrf_scoring_and_ranking(self) -> None:
        retriever = HybridRetriever(rrf_k=60)

        # Chunk A is rank 1 in BM25, rank 2 in Dense
        # Chunk B is rank 2 in BM25, rank 1 in Dense
        # Chunk C is rank 3 in BM25, not in Dense
        # Chunk D is not in BM25, rank 3 in Dense
        bm25_chunks = [
            make_chunk("A", "doc-A", score=10.0, retriever_name="bm25", rank=1),
            make_chunk("B", "doc-B", score=8.0, retriever_name="bm25", rank=2),
            make_chunk("C", "doc-C", score=5.0, retriever_name="bm25", rank=3),
        ]
        dense_chunks = [
            make_chunk("B", "doc-B", score=0.95, retriever_name="dense", rank=1),
            make_chunk("A", "doc-A", score=0.90, retriever_name="dense", rank=2),
            make_chunk("D", "doc-D", score=0.75, retriever_name="dense", rank=3),
        ]

        fused = retriever.reciprocal_rank_fusion(
            bm25_chunks=bm25_chunks,
            dense_chunks=dense_chunks,
            alpha=0.5,
            k=60,
        )

        assert len(fused) == 4
        # At alpha=0.5, w_bm25=1.0 and w_dense=1.0:
        # A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
        # B: 1/(60+2) + 1/(60+1) = 0.032522
        # Both A and B are top-2, significantly higher than C or D (which only have one contribution ~0.01587)
        top_ids = {fused[0].chunk_id, fused[1].chunk_id}
        assert top_ids == {"A", "B"}

        # Check attribution fields are populated
        chunk_a = next(c for c in fused if c.chunk_id == "A")
        assert chunk_a.bm25_rank == 1
        assert chunk_a.dense_rank == 2
        assert chunk_a.bm25_score == 10.0
        assert chunk_a.dense_score == 0.90
        assert chunk_a.retriever_name == "hybrid"

    def test_rrf_alpha_weighting(self) -> None:
        retriever = HybridRetriever(rrf_k=60)
        bm25_chunks = [make_chunk("A", "doc-A", score=10.0, rank=1)]
        dense_chunks = [make_chunk("B", "doc-B", score=0.95, rank=1)]

        # Alpha = 0.9 (Heavy weight to Dense)
        fused_dense_heavy = retriever.reciprocal_rank_fusion(
            bm25_chunks=bm25_chunks,
            dense_chunks=dense_chunks,
            alpha=0.9,
            k=60,
        )
        assert fused_dense_heavy[0].chunk_id == "B"

        # Alpha = 0.1 (Heavy weight to BM25)
        fused_bm25_heavy = retriever.reciprocal_rank_fusion(
            bm25_chunks=bm25_chunks,
            dense_chunks=dense_chunks,
            alpha=0.1,
            k=60,
        )
        assert fused_bm25_heavy[0].chunk_id == "A"


class TestRelativeScoreFusion:
    def test_rsf_min_max_normalization(self) -> None:
        retriever = HybridRetriever()
        bm25_chunks = [
            make_chunk("A", "doc-A", score=10.0, rank=1),
            make_chunk("B", "doc-B", score=0.0, rank=2),
        ]
        dense_chunks = [
            make_chunk("A", "doc-A", score=0.5, rank=2),
            make_chunk("B", "doc-B", score=1.0, rank=1),
        ]

        # Normalized BM25: A=1.0, B=0.0
        # Normalized Dense: A=0.0, B=1.0
        # With alpha=0.5:
        # A: 0.5*1.0 + 0.5*0.0 = 0.5
        # B: 0.5*0.0 + 0.5*1.0 = 0.5
        fused = retriever.relative_score_fusion(
            bm25_chunks=bm25_chunks,
            dense_chunks=dense_chunks,
            alpha=0.5,
        )
        assert len(fused) == 2
        assert fused[0].score == 0.5
        assert fused[1].score == 0.5


@pytest.mark.asyncio
class TestHybridRetrieverAsyncSearch:
    async def test_hybrid_search_end_to_end(self) -> None:
        mock_bm25 = AsyncMock()
        mock_bm25.search.return_value = RetrievalResult(
            query_text="RAG vs Dense",
            retriever="bm25",
            chunks=[make_chunk("A", "doc-A", score=5.0, rank=1)],
            total_found=1,
            latency_ms=12.0,
        )

        mock_dense = AsyncMock()
        mock_dense.search.return_value = RetrievalResult(
            query_text="RAG vs Dense",
            retriever="dense",
            chunks=[make_chunk("A", "doc-A", score=0.9, rank=1)],
            total_found=1,
            latency_ms=18.0,
        )

        hybrid = HybridRetriever(
            bm25_retriever=mock_bm25,
            dense_retriever=mock_dense,
            rrf_k=60,
        )
        result = await hybrid.search(RetrievalQuery(query_text="RAG vs Dense", top_k=5))

        assert result.retriever == "hybrid"
        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == "A"
        assert result.chunks[0].bm25_rank == 1
        assert result.chunks[0].dense_rank == 1
        assert "fusion_method" in result.metadata

"""
Unit tests for Passage Chunker (retrieval/passage_chunker.py).
"""
from __future__ import annotations

import pytest

from retrieval.passage_chunker import PassageChunker


class TestPassageChunker:
    def test_chunker_initialization(self) -> None:
        chunker = PassageChunker(chunk_size_tokens=256, chunk_overlap_tokens=30)
        assert chunker.chunk_size_tokens == 256
        assert chunker.chunk_overlap_tokens == 30
        assert chunker.max_chunk_chars == int(256 * 4.0)

    def test_chunker_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="must be less than"):
            PassageChunker(chunk_size_tokens=100, chunk_overlap_tokens=100)

        with pytest.raises(ValueError, match="must be less than"):
            PassageChunker(chunk_size_tokens=100, chunk_overlap_tokens=150)

    def test_chunk_empty_text_returns_empty_list(self) -> None:
        chunker = PassageChunker()
        chunks = chunker.chunk_document(document_id="doc1", content="")
        assert chunks == []

        chunks_ws = chunker.chunk_document(document_id="doc1", content="   \n\n  ")
        assert chunks_ws == []

    def test_chunk_short_text_single_chunk(self) -> None:
        chunker = PassageChunker(chunk_size_tokens=512, chunk_overlap_tokens=50)
        text = "This is a short research paper abstract. It discusses RAG pipelines and vector search."
        chunks = chunker.chunk_document(
            document_id="doc-short",
            content=text,
            source_type="arxiv",
            source_id="2005.12345",
            title="Short Paper",
        )
        assert len(chunks) == 1
        c = chunks[0]
        assert c.document_id == "doc-short"
        assert c.source_type == "arxiv"
        assert c.title == "Short Paper"
        assert c.chunk_index == 0
        assert c.total_chunks == 1
        assert "RAG pipelines" in c.content

    def test_sentence_boundary_preservation(self) -> None:
        chunker = PassageChunker(chunk_size_tokens=20, chunk_overlap_tokens=5)  # small chunks
        text = (
            "First sentence introduces the core problem. "
            "Second sentence explains the methodology. "
            "Third sentence details the empirical benchmark results. "
            "Fourth sentence concludes the findings."
        )
        chunks = chunker.chunk_document(
            document_id="doc-sent",
            content=text,
            title="Boundary Test",
        )
        assert len(chunks) >= 2
        for c in chunks:
            # Check that chunks don't start with broken sentence fragments in middle of words
            assert c.content[0].isupper() or c.content[0] == '"'
            assert c.total_chunks == len(chunks)

    def test_character_offset_tracking(self) -> None:
        chunker = PassageChunker(chunk_size_tokens=30, chunk_overlap_tokens=5)
        text = "Alpha is the first letter. Beta is the second letter. Gamma is the third letter."
        chunks = chunker.chunk_document(document_id="doc-offset", content=text)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.start_char >= 0
            assert c.end_char <= len(text)
            assert c.start_char <= c.end_char

    def test_metadata_copied_to_each_chunk(self) -> None:
        chunker = PassageChunker(chunk_size_tokens=25, chunk_overlap_tokens=5)
        meta = {"authors": ["Alice", "Bob"], "year": 2024}
        text = (
            "Sentence one discusses dense retrieval in depth. "
            "Sentence two discusses sparse retrieval in depth. "
            "Sentence three discusses cross-encoders."
        )
        chunks = chunker.chunk_document(
            document_id="doc-meta",
            content=text,
            metadata=meta,
        )
        assert len(chunks) >= 1
        for c in chunks:
            assert c.metadata["year"] == 2024
            assert c.metadata["authors"] == ["Alice", "Bob"]

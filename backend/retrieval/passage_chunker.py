"""
Passage chunker for breaking long documents into retrieval-ready passages.

Features:
- Sentence-aware boundary splitting (doesn't chop in the middle of a sentence)
- Token estimation (approx 4 chars per token)
- Configurable chunk size and overlap
- Character offset tracking (start_char, end_char)
- Preserves full document metadata on each chunk
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from retrieval.models import DocumentChunk


class PassageChunker:
    """
    Splits text into overlapping passages while respecting sentence boundaries.
    """

    # Sentence boundary regex (handles ., !, ?, followed by whitespace or quotes)
    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

    def __init__(
        self,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 50,
        chars_per_token: float = 4.0,
    ) -> None:
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({chunk_overlap_tokens}) must be less than "
                f"chunk_size_tokens ({chunk_size_tokens})"
            )
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.chars_per_token = chars_per_token
        
        self.max_chunk_chars = int(chunk_size_tokens * chars_per_token)
        self.overlap_chars = int(chunk_overlap_tokens * chars_per_token)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count based on character length and whitespace."""
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))

    def _split_into_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """
        Split text into sentences with start and end character positions.
        Returns list of (sentence_text, start_char, end_char).
        """
        if not text:
            return []

        sentences: list[tuple[str, int, int]] = []
        # Find paragraphs first
        paragraphs = text.split("\n\n")
        current_offset = 0

        for para in paragraphs:
            if not para.strip():
                current_offset += len(para) + 2
                continue

            # Split paragraph into sentences
            raw_sents = self.SENTENCE_SPLIT_REGEX.split(para)
            para_offset = 0

            for s in raw_sents:
                s_clean = s.strip()
                if not s_clean:
                    continue
                # Locate sentence in paragraph
                s_pos = para.find(s, para_offset)
                if s_pos == -1:
                    s_pos = para_offset
                start = current_offset + s_pos
                end = start + len(s)
                sentences.append((s_clean, start, end))
                para_offset = s_pos + len(s)

            current_offset += len(para) + 2  # +2 for \n\n

        if not sentences and text.strip():
            sentences.append((text.strip(), 0, len(text)))

        return sentences

    def chunk_document(
        self,
        document_id: str,
        content: str,
        source_type: str = "local",
        source_id: str = "",
        title: str = "",
        url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk a document into a sequence of DocumentChunk objects.
        """
        clean_content = content.strip()
        if not clean_content:
            return []

        meta = metadata or {}
        sentences = self._split_into_sentences(clean_content)
        if not sentences:
            return []

        chunks: list[DocumentChunk] = []
        current_sentences: list[str] = []
        current_length = 0
        chunk_start_char = sentences[0][1]
        chunk_end_char = sentences[0][2]
        i = 0

        while i < len(sentences):
            sent_text, s_start, s_end = sentences[i]
            sent_len = len(sent_text) + 1  # include space

            # If a single sentence exceeds max_chunk_chars, we must break it down by chars
            if sent_len > self.max_chunk_chars and not current_sentences:
                # Force chunk long sentence
                for sub_start in range(0, len(sent_text), self.max_chunk_chars - self.overlap_chars):
                    sub_text = sent_text[sub_start : sub_start + self.max_chunk_chars]
                    if sub_text:
                        chunks.append(
                            DocumentChunk(
                                id=str(uuid.uuid4()),
                                document_id=document_id,
                                source_type=source_type,
                                source_id=source_id,
                                title=title,
                                url=url,
                                content=sub_text,
                                chunk_index=len(chunks),
                                total_chunks=0,  # updated later
                                start_char=s_start + sub_start,
                                end_char=s_start + sub_start + len(sub_text),
                                metadata=meta.copy(),
                            )
                        )
                i += 1
                if i < len(sentences):
                    chunk_start_char = sentences[i][1]
                continue

            if current_length + sent_len <= self.max_chunk_chars or not current_sentences:
                current_sentences.append(sent_text)
                current_length += sent_len
                chunk_end_char = s_end
                i += 1
            else:
                # Emit current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        document_id=document_id,
                        source_type=source_type,
                        source_id=source_id,
                        title=title,
                        url=url,
                        content=chunk_text,
                        chunk_index=len(chunks),
                        total_chunks=0,  # updated later
                        start_char=chunk_start_char,
                        end_char=chunk_end_char,
                        metadata=meta.copy(),
                    )
                )

                # Overlap step: keep trailing sentences that fit within overlap_chars
                overlap_acc: list[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) + 1 <= self.overlap_chars:
                        overlap_acc.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break

                current_sentences = overlap_acc
                current_length = overlap_len
                # Next chunk start will be from the beginning of retained sentences or current sentence
                if overlap_acc:
                    # Look up start of first retained sentence
                    for s_item, ss_start, _ in sentences:
                        if s_item == overlap_acc[0]:
                            chunk_start_char = ss_start
                            break
                else:
                    chunk_start_char = s_start

        # Flush final chunk if any
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    source_type=source_type,
                    source_id=source_id,
                    title=title,
                    url=url,
                    content=chunk_text,
                    chunk_index=len(chunks),
                    total_chunks=0,
                    start_char=chunk_start_char,
                    end_char=chunk_end_char,
                    metadata=meta.copy(),
                )
            )

        # Set total_chunks on all chunks
        total = len(chunks)
        for c in chunks:
            c.total_chunks = total

        return chunks

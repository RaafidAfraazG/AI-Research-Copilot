"""
Local BM25 search engine for local development and offline testing.
Implements Okapi BM25 with field boosting and persistence.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any
import structlog

from retrieval.models import DocumentChunk, RetrievalQuery, ScoredChunk

logger = structlog.get_logger(__name__)


class LocalBM25Store:
    """
    Self-contained in-memory / file-persisted Okapi BM25 index for local development.
    Preserves exact compatibility with OpenSearch BM25 results.
    """

    def __init__(
        self,
        storage_file: Path | str | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.storage_file = Path(storage_file) if storage_file else None
        self.chunks: dict[str, DocumentChunk] = {}
        self._doc_lens: dict[str, int] = {}
        self._tf: dict[str, dict[str, int]] = {}  # chunk_id -> term -> count
        self._df: dict[str, int] = {}  # term -> document frequency
        self._avg_dl: float = 0.0
        self.index_generation: str | None = None

        if self.storage_file and self.storage_file.exists():
            self._load()

    def _tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def remove_document(self, document_id: str, save: bool = True) -> int:
        """Remove every chunk of a document. Returns the number of chunks removed."""
        ids = [cid for cid, c in self.chunks.items() if c.document_id == document_id]
        for cid in ids:
            self.chunks.pop(cid, None)
            self._doc_lens.pop(cid, None)
            self._tf.pop(cid, None)
        if ids:
            self._recompute_stats()
            if save and self.storage_file:
                self._save()
        return len(ids)

    def add_chunks(self, chunks: list[DocumentChunk], save: bool = True) -> None:
        """Index a batch of DocumentChunks (re-adding an existing chunk id replaces it)."""
        for c in chunks:
            self.chunks[c.id] = c
            # Weighted token stream: title x2 + abstract x2 + content x1
            title_tokens = self._tokenize(c.title) * 2
            meta_abstract = c.metadata.get("abstract", "") if c.metadata else ""
            abstract_tokens = self._tokenize(meta_abstract) * 2
            content_tokens = self._tokenize(c.content)

            all_tokens = title_tokens + abstract_tokens + content_tokens
            doc_len = len(all_tokens)
            self._doc_lens[c.id] = doc_len

            term_counts: dict[str, int] = {}
            for t in all_tokens:
                term_counts[t] = term_counts.get(t, 0) + 1
            self._tf[c.id] = term_counts

        self._recompute_stats()

        if save and self.storage_file:
            self._save()

    def _recompute_stats(self) -> None:
        # Recompute document frequencies and avg_dl
        self._df.clear()
        for doc_id, tf_map in self._tf.items():
            for term in tf_map:
                self._df[term] = self._df.get(term, 0) + 1

        total_tokens = sum(self._doc_lens.values())
        total_docs = len(self.chunks)
        self._avg_dl = total_tokens / total_docs if total_docs > 0 else 1.0

    def search(self, query: RetrievalQuery) -> list[ScoredChunk]:
        """Search the local BM25 index."""
        q_tokens = self._tokenize(query.query_text)
        if not q_tokens or not self.chunks:
            return []

        n_docs = len(self.chunks)
        scores: dict[str, float] = {}

        for t in q_tokens:
            if t not in self._df:
                continue
            df = self._df[t]
            # Lucene / OpenSearch BM25 IDF variant
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

            for chunk_id, tf_map in self._tf.items():
                if t not in tf_map:
                    continue
                tf = tf_map[t]
                doc_len = self._doc_lens.get(chunk_id, 1)
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self._avg_dl)))
                score = idf * (num / denom)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        # Apply source type and metadata filters
        filtered_scores: list[tuple[str, float]] = []
        for chunk_id, score in scores.items():
            chunk = self.chunks[chunk_id]
            if query.source_types and chunk.source_type not in query.source_types:
                continue
            if query.filters:
                match = True
                for k, v in query.filters.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            filtered_scores.append((chunk_id, score))

        # Sort descending by score
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        top_hits = filtered_scores[: query.top_k]

        scored_chunks: list[ScoredChunk] = []
        for rank, (cid, score) in enumerate(top_hits, start=1):
            c = self.chunks[cid]
            scored_chunks.append(
                ScoredChunk(
                    chunk_id=c.id,
                    document_id=c.document_id,
                    source_type=c.source_type,
                    source_id=c.source_id,
                    title=c.title,
                    url=c.url,
                    content=c.content,
                    chunk_index=c.chunk_index,
                    total_chunks=c.total_chunks,
                    score=round(score, 4),
                    retriever_name="bm25",
                    rank=rank,
                    bm25_score=round(score, 4),
                    bm25_rank=rank,
                    metadata=c.metadata,
                )
            )
        return scored_chunks

    def _save(self) -> None:
        if not self.storage_file:
            return
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        # Embeddings live in Qdrant only; the BM25 file is purely lexical.
        data = {
            "index_generation": self.index_generation,
            "chunks": [c.model_dump(exclude={"embedding"}) for c in self.chunks.values()],
        }
        tmp = self.storage_file.with_name(self.storage_file.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.storage_file)

    def _load(self) -> None:
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.index_generation = data.get("index_generation")
            chunks = [DocumentChunk(**item) for item in data.get("chunks", [])]
            self.add_chunks(chunks, save=False)
        except Exception as e:
            logger.warning("failed_to_load_local_bm25_store", error=str(e))

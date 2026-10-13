"""
Data models and schemas for the retrieval pipeline.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """
    A single text chunk from an indexed document, ready for indexing or ranking.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    source_type: str = "local"
    source_id: str
    title: str = ""
    url: str | None = None
    content: str
    chunk_index: int = 0
    total_chunks: int = 1
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class ScoredChunk(BaseModel):
    """
    A document chunk with associated relevance score and retrieval metadata.
    """
    chunk_id: str
    document_id: str
    source_type: str
    source_id: str
    title: str = ""
    url: str | None = None
    content: str
    chunk_index: int = 0
    total_chunks: int = 1
    
    # Relevance scoring
    score: float = 0.0
    retriever_name: str = "hybrid"  # "bm25", "dense", "hybrid"
    rank: int | None = None
    
    # Detailed attribution
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None
    
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    """
    Structured query sent to retrievers.
    """
    query_text: str
    top_k: int = Field(default=10, ge=1, le=100)
    source_types: list[str] | None = None
    filters: dict[str, Any] | None = None
    min_score: float | None = None
    alpha: float = 0.5  # Weight balance: 0.0 = BM25 only, 1.0 = Dense only, 0.5 = Balanced


class RetrievalResult(BaseModel):
    """
    Aggregated response returned by any retriever.
    """
    query_text: str
    retriever: str
    chunks: list[ScoredChunk]
    total_found: int
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestDocumentRequest(BaseModel):
    """
    Request model for ingesting a document into the retrieval pipeline.
    """
    source_type: str = "local"
    source_id: str
    title: str
    content: str
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_size_tokens: int | None = None
    chunk_overlap_tokens: int | None = None


class IngestDocumentResponse(BaseModel):
    """
    Response after ingesting and indexing a document.
    """
    document_id: str
    source_type: str
    source_id: str
    total_chunks: int
    indexed_in_opensearch: bool
    indexed_in_qdrant: bool
    status: str = "success"
    message: str = "Document successfully indexed"

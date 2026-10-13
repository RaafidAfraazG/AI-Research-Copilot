"""
API endpoints for search, retrieval, document indexing, and passage chunking.
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_hybrid_retriever, get_indexing_service
from core.exceptions import AppException
from db.database import get_db_session
from embedding.factory import get_embedding_provider
from retrieval.index_manifest import IndexStore
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import (
    DocumentChunk,
    IngestDocumentRequest,
    IngestDocumentResponse,
    RetrievalQuery,
    RetrievalResult,
    ScoredChunk,
)
from retrieval.passage_chunker import PassageChunker

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class SearchRequest(BaseModel):
    """Search request body."""
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of passages to return")
    mode: Literal["hybrid", "bm25", "dense"] = Field(
        default="hybrid",
        description="Retrieval engine mode",
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weighting balance for hybrid search: 0.0=BM25, 1.0=Dense, 0.5=Equal",
    )
    fusion_method: Literal["rrf", "rsf"] = Field(
        default="rrf",
        description="Fusion algorithm for hybrid retrieval (RRF: Reciprocal Rank Fusion, RSF: Relative Score Fusion)",
    )
    source_types: list[str] | None = Field(
        default=None,
        description="Filter results by source type(s), e.g. ['arxiv', 'web']",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary metadata filters",
    )


class ChunkPreviewRequest(BaseModel):
    """Text chunking preview request."""
    text: str = Field(..., min_length=1)
    chunk_size_tokens: int = Field(default=512, ge=50, le=4096)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=1024)
    title: str = "Preview Document"


class ChunkPreviewResponse(BaseModel):
    """Text chunking preview response."""
    total_chunks: int
    chunks: list[DocumentChunk]


@router.post(
    "/search",
    response_model=RetrievalResult,
    summary="Execute multi-mode document retrieval",
    description="Retrieve document passages using BM25, Dense Vector, or Hybrid (RRF) search.",
)
async def search_documents(request: SearchRequest) -> RetrievalResult:
    """Execute search across indexed passages."""
    retrieval_query = RetrievalQuery(
        query_text=request.query,
        top_k=request.top_k,
        source_types=request.source_types,
        filters=request.filters,
        alpha=request.alpha,
    )

    hybrid: HybridRetriever = get_hybrid_retriever()
    if request.mode == "bm25":
        return await hybrid.bm25_retriever.search(retrieval_query)
    elif request.mode == "dense":
        return await hybrid.dense_retriever.search(retrieval_query)
    else:
        retriever = hybrid
        return await retriever.search(
            query=retrieval_query,
            fusion_method=request.fusion_method,
        )


@router.post(
    "/index",
    response_model=IngestDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest and index a document",
    description="Chunk, embed, and index a document into both OpenSearch and Qdrant.",
)
async def index_document(
    request: IngestDocumentRequest,
    db: AsyncSession = Depends(get_db_session),
) -> IngestDocumentResponse:
    """Ingest a document into the search indices."""
    service = get_indexing_service()
    return await service.index_document(request, db_session=db)


@router.post(
    "/chunk",
    response_model=ChunkPreviewResponse,
    summary="Preview passage chunking",
    description="Preview how a text document is split into sentence-bounded overlapping chunks.",
)
async def preview_chunking(request: ChunkPreviewRequest) -> ChunkPreviewResponse:
    """Preview passage chunking for arbitrary text."""
    if request.chunk_overlap_tokens >= request.chunk_size_tokens:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="chunk_overlap_tokens must be strictly less than chunk_size_tokens",
        )

    chunker = PassageChunker(
        chunk_size_tokens=request.chunk_size_tokens,
        chunk_overlap_tokens=request.chunk_overlap_tokens,
    )
    chunks = chunker.chunk_document(
        document_id="preview",
        content=request.text,
        title=request.title,
    )
    return ChunkPreviewResponse(
        total_chunks=len(chunks),
        chunks=chunks,
    )


@router.get(
    "/index/status",
    summary="Active index and embedding diagnostics",
    description=(
        "Reports the active index generation, embedding model identity, dimension, counts "
        "and whether the index is compatible with the configured embedding provider. "
        "Exposes no filesystem paths or secrets."
    ),
)
async def index_status() -> dict[str, Any]:
    # Describing the provider may load the embedding model (cold: imports torch, tens of
    # seconds). Do it in a worker thread so /health and other requests are not blocked.
    def _compute() -> dict[str, Any]:
        store = IndexStore()
        try:
            provider = get_embedding_provider()
            provider.model_name  # property access only
        except AppException as e:
            info = store.diagnostics(None)
            info["embedding_provider_error"] = e.code
            return info
        try:
            return store.diagnostics(provider)
        except AppException as e:  # model failed to load while describing the provider
            info = store.diagnostics(None)
            info["embedding_provider_error"] = e.code
            info["detail"] = e.message
            return info

    return await asyncio.to_thread(_compute)

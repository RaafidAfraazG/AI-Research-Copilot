"""Retrieval module for AI Research Copilot."""
from retrieval.base import BaseRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.indexing_service import IndexingService
from retrieval.models import (
    DocumentChunk,
    IngestDocumentRequest,
    IngestDocumentResponse,
    RetrievalQuery,
    RetrievalResult,
    ScoredChunk,
)
from retrieval.passage_chunker import PassageChunker

__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "IndexingService",
    "PassageChunker",
    "DocumentChunk",
    "ScoredChunk",
    "RetrievalQuery",
    "RetrievalResult",
    "IngestDocumentRequest",
    "IngestDocumentResponse",
]

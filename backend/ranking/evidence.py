"""
Typed evidence data structures and schemas for evidence packaging.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class EvidencePassage(BaseModel):
    """
    A single selected, deduplicated, and ranked evidence passage.
    """
    passage_id: str = Field(..., description="Unique chunk/passage identifier")
    document_id: str = Field(..., description="Unique parent document identifier")
    text: str = Field(..., description="Passage body text")
    title: str = Field(default="", description="Parent document title")
    source_type: str = Field(..., description="Source type (arxiv, semantic_scholar, web, local)")
    source_id: str = Field(..., description="Native source identifier (e.g. arXiv ID, S2 ID, URL)")
    url: str | None = Field(default=None, description="Direct URL to source document")
    relevance_score: float = Field(..., description="Cross-encoder relevance or final ranking score")
    rank: int = Field(..., description="1-indexed rank within selected evidence set")
    chunk_index: int = Field(default=0, description="Index of this chunk within parent document")
    provenance: dict[str, Any] = Field(default_factory=dict, description="Source attribution metadata")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional passage metadata")


class EvidencePack(BaseModel):
    """
    Structured collection of verified, deduplicated evidence passages
    ready for consumption by the LLM synthesis layer.
    """
    query: str = Field(..., description="Original user research query")
    selected_passages: list[EvidencePassage] = Field(
        default_factory=list,
        description="Top-K deduplicated, diversified evidence passages",
    )
    total_candidates: int = Field(default=0, description="Total candidate passages retrieved before reranking")
    total_evidence: int = Field(default=0, description="Number of evidence passages selected")
    retrieval_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata from lexical, dense, and hybrid retrieval stages",
    )
    ranking_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata from the cross-encoder reranker (model, latency, device)",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregated source provenance across all selected passages",
    )

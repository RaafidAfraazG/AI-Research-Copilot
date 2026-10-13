"""
Normalized data models and schemas for research documents and source connectors.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any
import uuid
from pydantic import BaseModel, Field

from retrieval.models import IngestDocumentRequest


class CitationInfo(BaseModel):
    """Citation and reference metrics for a document."""
    citation_count: int | None = None
    reference_count: int | None = None
    influential_citation_count: int | None = None
    tldr: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    corpus_id: str | None = None


class ProvenanceInfo(BaseModel):
    """Audit and provenance metadata tracking document origin."""
    source_type: str
    source_id: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    canonical_url: str | None = None
    content_hash: str = ""
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    cross_source_ids: dict[str, str] = Field(default_factory=dict)


class ResearchDocument(BaseModel):
    """
    Canonical normalized document model across all research sources
    (arXiv, Semantic Scholar, Web, etc.).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: str  # "arxiv", "semantic_scholar", "web", "local"
    source_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    full_text: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    publication_date: str | None = None
    venue: str | None = None
    categories: list[str] = Field(default_factory=list)
    citation_info: CitationInfo = Field(default_factory=CitationInfo)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: ProvenanceInfo | None = None

    def compute_content_hash(self) -> str:
        """Compute SHA-256 hash of the document content."""
        content = self.get_indexable_content()
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_indexable_content(self) -> str:
        """
        Combine abstract and full text into a unified string for chunking and retrieval.
        """
        parts: list[str] = []
        if self.abstract:
            parts.append(self.abstract.strip())
        if self.full_text and self.full_text.strip() != (self.abstract or "").strip():
            parts.append(self.full_text.strip())
        return "\n\n".join(parts) if parts else self.title

    def to_ingest_request(
        self,
        chunk_size_tokens: int | None = None,
        chunk_overlap_tokens: int | None = None,
    ) -> IngestDocumentRequest:
        """
        Convert to an IngestDocumentRequest for the retrieval indexing pipeline.
        """
        meta = self.metadata.copy()
        meta.update({
            "authors": self.authors,
            "publication_date": self.publication_date,
            "venue": self.venue,
            "categories": self.categories,
            "citation_count": self.citation_info.citation_count,
            "tldr": self.citation_info.tldr,
            "doi": self.citation_info.doi,
            "pdf_url": self.pdf_url,
        })
        if self.abstract:
            meta["abstract"] = self.abstract

        return IngestDocumentRequest(
            source_type=self.source_type,
            source_id=self.source_id,
            title=self.title,
            content=self.get_indexable_content(),
            url=self.url,
            metadata=meta,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )


class ConnectorSearchRequest(BaseModel):
    """Search request parameters for a source connector."""
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
    offset: int = Field(default=0, ge=0)
    year: int | None = None
    categories: list[str] | None = None
    sort_by: str = "relevance"


class ConnectorSearchResponse(BaseModel):
    """Aggregated search results returned by a source connector."""
    source_type: str
    query: str
    total_results: int
    documents: list[ResearchDocument]
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

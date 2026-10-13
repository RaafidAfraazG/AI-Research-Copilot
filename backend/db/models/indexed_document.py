"""IndexedDocument ORM model."""
from __future__ import annotations

import uuid

from sqlalchemy import JSON, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from db.models.base import Base, TimestampMixin


class DocumentSource:
    """Known document source types for the source_type column."""
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    WEB = "web"
    GITHUB = "github"
    LOCAL = "local"


class IndexedDocument(Base, TimestampMixin):
    """
    Registry of documents known to the search system.

    This table tracks metadata about documents that have been ingested into
    OpenSearch (for BM25 search) and Qdrant (for dense vector search).

    It does NOT store the full document content — that lives in OpenSearch.
    The content_hash allows detection of changes to avoid re-indexing unchanged documents.

    The (source_type, source_id) pair is unique — the natural key for a document
    from a given source (e.g., source_type="arxiv", source_id="2304.00123").
    """

    __tablename__ = "indexed_documents"
    __table_args__ = (
        UniqueConstraint("source_type", "source_id", name="uq_indexed_documents_source"),
        Index("ix_indexed_documents_source_type", "source_type"),
        Index("ix_indexed_documents_content_hash", "content_hash"),
        Index("ix_indexed_documents_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    # Source identification — see DocumentSource constants
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_id: Mapped[str] = mapped_column(String(500), nullable=False)

    # Document metadata
    title: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # SHA-256 hash of the document content — used to skip re-indexing unchanged docs
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Arbitrary JSON: authors, publication date, abstract, tags, etc.
    extra_metadata: Mapped[dict | None] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
        default=None,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<IndexedDocument id={self.id} source={self.source_type}:{self.source_id!r}>"

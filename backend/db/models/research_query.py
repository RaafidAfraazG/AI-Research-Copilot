"""ResearchQuery ORM model."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import JSON, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from db.models.research_session import ResearchSession


class QueryStatus:
    """Enumerated status values for a research query lifecycle."""
    PENDING = "pending"        # Accepted, not yet processed
    PROCESSING = "processing"  # Pipeline currently running
    COMPLETED = "completed"    # Pipeline finished successfully
    FAILED = "failed"          # Pipeline encountered an unrecoverable error


class ResearchQuery(Base, TimestampMixin):
    """
    Represents a single research query submitted by the user.

    A query belongs to an optional session and tracks the full lifecycle
    from submission through retrieval and synthesis to completion.
    """

    __tablename__ = "research_queries"
    __table_args__ = (
        Index("ix_research_queries_session_id", "session_id"),
        Index("ix_research_queries_status", "status"),
        Index("ix_research_queries_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    # FK to session — nullable so queries can exist without a session
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )

    # The raw query text as entered by the user
    query_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Cleaned/normalised form of the query (lower-cased, stripped, etc.)
    # Populated by the query intelligence service
    normalized_query: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Lifecycle status — see QueryStatus constants above
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=QueryStatus.PENDING,
    )

    # Arbitrary JSON: sub-queries, intent, latency breakdown, error details, etc.
    extra_metadata: Mapped[dict | None] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
        default=None,
    )

    # Relationships
    session: Mapped[ResearchSession | None] = relationship(
        "ResearchSession",
        back_populates="queries",
        lazy="select",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ResearchQuery id={self.id} status={self.status!r}>"

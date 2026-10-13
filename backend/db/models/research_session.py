"""ResearchSession ORM model."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from db.models.research_query import ResearchQuery


class ResearchSession(Base, TimestampMixin):
    """
    Represents a user's research session — a logical grouping of related queries.

    A session might correspond to a single research topic or investigation thread.
    Sessions are optional: a ResearchQuery can exist without a parent session.
    """

    __tablename__ = "research_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Arbitrary JSON metadata (tags, user preferences, etc.)
    # Named 'extra_metadata' in Python to avoid conflict with SQLAlchemy's .metadata attribute
    extra_metadata: Mapped[dict | None] = mapped_column(
        "metadata",    # column name in DB is still 'metadata'
        JSON,
        nullable=True,
        default=None,
    )

    # Relationships
    queries: Mapped[list[ResearchQuery]] = relationship(
        "ResearchQuery",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ResearchSession id={self.id} title={self.title!r}>"

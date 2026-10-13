"""
ORM models package.

Importing this package ensures all models are registered with Base.metadata,
which is required for Alembic's autogenerate to detect all tables.
"""
from db.models.base import Base, TimestampMixin
from db.models.indexed_document import DocumentSource, IndexedDocument
from db.models.research_query import QueryStatus, ResearchQuery
from db.models.research_session import ResearchSession

__all__ = [
    "Base",
    "TimestampMixin",
    "ResearchSession",
    "ResearchQuery",
    "QueryStatus",
    "IndexedDocument",
    "DocumentSource",
]

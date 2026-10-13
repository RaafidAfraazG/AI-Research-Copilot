"""
Tests for ORM models using an in-memory SQLite database.

These tests verify:
- Model instances can be created and flushed
- Primary key (UUID) is auto-generated
- Timestamps are set on creation
- Relationships (session → queries) work correctly
- Constraints (UniqueConstraint on IndexedDocument) are enforced

Tests use the `db_session` fixture from conftest.py which provides
a SQLite-backed async session. No Docker or PostgreSQL is required.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession


class TestResearchSession:
    async def test_create_minimal_session(self, db_session: AsyncSession) -> None:
        from db.models.research_session import ResearchSession

        session = ResearchSession()
        db_session.add(session)
        await db_session.flush()

        assert session.id is not None
        assert isinstance(session.id, uuid.UUID)
        assert session.created_at is not None
        assert session.updated_at is not None

    async def test_create_session_with_title(self, db_session: AsyncSession) -> None:
        from db.models.research_session import ResearchSession

        session = ResearchSession(title="Edge AI Research 2024")
        db_session.add(session)
        await db_session.flush()

        assert session.title == "Edge AI Research 2024"

    async def test_session_with_metadata(self, db_session: AsyncSession) -> None:
        from db.models.research_session import ResearchSession

        meta = {"tags": ["ml", "edge-ai"], "priority": "high"}
        session = ResearchSession(title="ML Research", extra_metadata=meta)
        db_session.add(session)
        await db_session.flush()

        assert session.extra_metadata == meta

    async def test_two_sessions_have_different_ids(self, db_session: AsyncSession) -> None:
        from db.models.research_session import ResearchSession

        s1 = ResearchSession(title="Session A")
        s2 = ResearchSession(title="Session B")
        db_session.add_all([s1, s2])
        await db_session.flush()

        assert s1.id != s2.id


class TestResearchQuery:
    async def test_create_minimal_query(self, db_session: AsyncSession) -> None:
        from db.models.research_query import QueryStatus, ResearchQuery

        query = ResearchQuery(query_text="Compare YOLO vs DETR for edge deployment")
        db_session.add(query)
        await db_session.flush()

        assert query.id is not None
        assert isinstance(query.id, uuid.UUID)
        assert query.query_text == "Compare YOLO vs DETR for edge deployment"
        assert query.status == QueryStatus.PENDING

    async def test_query_default_status_is_pending(self, db_session: AsyncSession) -> None:
        from db.models.research_query import QueryStatus, ResearchQuery

        query = ResearchQuery(query_text="What is RAG?")
        db_session.add(query)
        await db_session.flush()

        assert query.status == QueryStatus.PENDING

    async def test_query_linked_to_session(self, db_session: AsyncSession) -> None:
        from db.models.research_query import ResearchQuery
        from db.models.research_session import ResearchSession

        session = ResearchSession(title="Test Session")
        db_session.add(session)
        await db_session.flush()

        query = ResearchQuery(session_id=session.id, query_text="What is transformer?")
        db_session.add(query)
        await db_session.flush()

        assert query.session_id == session.id

    async def test_query_without_session_is_valid(self, db_session: AsyncSession) -> None:
        from db.models.research_query import ResearchQuery

        # session_id is nullable — standalone queries are allowed
        query = ResearchQuery(query_text="Standalone query")
        db_session.add(query)
        await db_session.flush()

        assert query.session_id is None

    async def test_query_with_metadata(self, db_session: AsyncSession) -> None:
        from db.models.research_query import ResearchQuery

        meta = {"sub_queries": ["q1", "q2"], "intent": "comparison"}
        query = ResearchQuery(query_text="Compare A vs B", extra_metadata=meta)
        db_session.add(query)
        await db_session.flush()

        assert query.extra_metadata["intent"] == "comparison"


class TestIndexedDocument:
    async def test_create_arxiv_document(self, db_session: AsyncSession) -> None:
        from db.models.indexed_document import DocumentSource, IndexedDocument

        doc = IndexedDocument(
            source_type=DocumentSource.ARXIV,
            source_id="2304.00123",
            title="YOLO: Real-Time Object Detection",
            url="https://arxiv.org/abs/2304.00123",
        )
        db_session.add(doc)
        await db_session.flush()

        assert doc.id is not None
        assert doc.source_type == "arxiv"
        assert doc.source_id == "2304.00123"

    async def test_document_source_constants(self) -> None:
        from db.models.indexed_document import DocumentSource

        assert DocumentSource.ARXIV == "arxiv"
        assert DocumentSource.SEMANTIC_SCHOLAR == "semantic_scholar"
        assert DocumentSource.WEB == "web"
        assert DocumentSource.GITHUB == "github"
        assert DocumentSource.LOCAL == "local"

    async def test_document_with_content_hash(self, db_session: AsyncSession) -> None:
        from db.models.indexed_document import DocumentSource, IndexedDocument

        doc = IndexedDocument(
            source_type=DocumentSource.WEB,
            source_id="https://example.com/article",
            content_hash="a" * 64,
        )
        db_session.add(doc)
        await db_session.flush()

        assert doc.content_hash == "a" * 64

    async def test_document_metadata(self, db_session: AsyncSession) -> None:
        from db.models.indexed_document import DocumentSource, IndexedDocument

        meta = {"authors": ["John Doe"], "year": 2024, "venue": "NeurIPS"}
        doc = IndexedDocument(
            source_type=DocumentSource.ARXIV,
            source_id="2401.99999",
            extra_metadata=meta,
        )
        db_session.add(doc)
        await db_session.flush()

        assert doc.extra_metadata["venue"] == "NeurIPS"

    async def test_unique_constraint_source_type_and_id(self, db_session: AsyncSession) -> None:
        """Two documents from the same source with the same ID should violate uniqueness."""
        from sqlalchemy.exc import IntegrityError

        from db.models.indexed_document import DocumentSource, IndexedDocument

        doc1 = IndexedDocument(source_type=DocumentSource.ARXIV, source_id="2304.00001")
        doc2 = IndexedDocument(source_type=DocumentSource.ARXIV, source_id="2304.00001")
        db_session.add(doc1)
        await db_session.flush()

        db_session.add(doc2)
        with pytest.raises(IntegrityError):
            await db_session.flush()

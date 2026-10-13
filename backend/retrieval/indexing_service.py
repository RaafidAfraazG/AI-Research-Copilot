"""
Indexing Service for chunking, embedding, and dual-indexing documents into
OpenSearch (BM25) and Qdrant (Dense Vectors).
"""
from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any
import structlog
from opensearchpy import AsyncOpenSearch
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings, get_settings
from db.models.indexed_document import IndexedDocument
from embedding.base import EmbeddingProvider
from embedding.factory import get_embedding_provider
from retrieval.index_manifest import (
    CHUNKER_VERSION,
    ChunkingConfig,
    IndexStore,
    deterministic_chunk_id,
    deterministic_document_id,
    validate_vectors,
)
from retrieval.models import DocumentChunk, IngestDocumentRequest, IngestDocumentResponse
from retrieval.opensearch_schema import ensure_opensearch_index
from retrieval.passage_chunker import PassageChunker
from retrieval.qdrant_schema import ensure_qdrant_collection

logger = structlog.get_logger(__name__)


class IndexingService:
    """
    Coordinates document processing:
    1. Chunking via PassageChunker
    2. Embedding generation via EmbeddingProvider
    3. Bulk indexing into OpenSearch for BM25
    4. Upserting points into Qdrant for dense search
    5. Registry tracking in PostgreSQL
    """

    def __init__(
        self,
        opensearch_client: AsyncOpenSearch | None = None,
        qdrant_client: AsyncQdrantClient | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        chunker: PassageChunker | None = None,
        settings: Settings | None = None,
        index_store: IndexStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.index_store = index_store or IndexStore(settings=self.settings)
        self.chunking_config = ChunkingConfig(
            chunk_size_tokens=self.settings.chunk_size_tokens,
            chunk_overlap_tokens=self.settings.chunk_overlap_tokens,
            chunker_version=CHUNKER_VERSION,
        )
        self.chunker = chunker or PassageChunker(
            chunk_size_tokens=self.settings.chunk_size_tokens,
            chunk_overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        self.embedding_provider = embedding_provider or get_embedding_provider()
        
        self.opensearch_client = opensearch_client or AsyncOpenSearch(
            hosts=[{"host": self.settings.opensearch_host, "port": self.settings.opensearch_port}],
            http_auth=(
                (self.settings.opensearch_user, self.settings.opensearch_password)
                if self.settings.opensearch_user and self.settings.opensearch_password
                else None
            ),
            use_ssl=self.settings.opensearch_use_ssl,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=self.settings.opensearch_timeout,
        )

        self.qdrant_client = qdrant_client or AsyncQdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            api_key=self.settings.qdrant_api_key,
            timeout=self.settings.qdrant_timeout,
        )

        self._opensearch_alive: bool | None = None
        self._qdrant_alive: bool | None = None

    async def aclose(self) -> None:
        """Close OpenSearch/Qdrant client sessions (idempotent)."""
        for client in (self.opensearch_client, self.qdrant_client):
            try:
                await client.close()
            except Exception as e:
                logger.debug("index_client_close_failed", error_type=type(e).__name__)

    async def initialize_stores(self, recreate: bool = False) -> None:
        """Initialize both OpenSearch index and Qdrant collection."""
        try:
            await ensure_opensearch_index(
                client=self.opensearch_client,
                index_name=self.settings.opensearch_index_name,
                recreate=recreate,
            )
            self._opensearch_alive = True
        except Exception:
            self._opensearch_alive = False

        try:
            await ensure_qdrant_collection(
                client=self.qdrant_client,
                collection_name=self.settings.qdrant_collection_name,
                vector_dimension=self.embedding_provider.dimension,
                recreate=recreate,
            )
            self._qdrant_alive = True
        except Exception:
            self._qdrant_alive = False

    async def index_document(
        self,
        request: IngestDocumentRequest,
        db_session: AsyncSession | None = None,
    ) -> IngestDocumentResponse:
        """
        Process and index a single document.
        """
        start_time = time.perf_counter()
        # Deterministic id: re-indexing the same (source_type, source_id) REPLACES chunks.
        doc_uuid = deterministic_document_id(request.source_type, request.source_id)
        content_hash = hashlib.sha256(request.content.encode("utf-8")).hexdigest()

        # Custom chunker if specific token sizes requested
        chunker = self.chunker
        if request.chunk_size_tokens and request.chunk_overlap_tokens:
            chunker = PassageChunker(
                chunk_size_tokens=request.chunk_size_tokens,
                chunk_overlap_tokens=request.chunk_overlap_tokens,
            )

        # 1. Chunk document
        chunks = chunker.chunk_document(
            document_id=doc_uuid,
            content=request.content,
            source_type=request.source_type,
            source_id=request.source_id,
            title=request.title,
            url=request.url,
            metadata=request.metadata,
        )

        if not chunks:
            return IngestDocumentResponse(
                document_id=doc_uuid,
                source_type=request.source_type,
                source_id=request.source_id,
                total_chunks=0,
                indexed_in_opensearch=False,
                indexed_in_qdrant=False,
                status="empty",
                message="Document contained no indexable text",
            )

        for c in chunks:
            c.id = deterministic_chunk_id(doc_uuid, c.chunk_index)

        # 2. Fail closed on missing/incompatible index BEFORE generating or writing vectors
        manifest = self.index_store.ensure_active_for_write(
            self.embedding_provider, self.chunking_config
        )

        # 3. Compute embeddings for all chunks in batch and validate them
        chunk_texts = [c.content for c in chunks]
        embeddings = await self.embedding_provider.aembed_texts(chunk_texts)
        validate_vectors(embeddings, manifest.embedding.dimension, manifest.embedding.normalized)
        for c, emb in zip(chunks, embeddings):
            c.embedding = emb

        # 4. Optional remote OpenSearch (bulk) and remote Qdrant, if a server is reachable
        os_remote_ok = await self._index_opensearch_remote(chunks)
        qd_remote_ok = await self._index_qdrant_remote(chunks, manifest.generation_id)

        # 5. Canonical local stores (BM25 JSON + embedded Qdrant), duplicate-safe
        local_ok = True
        try:
            self.index_store.upsert_document(manifest.generation_id, doc_uuid, chunks)
            self.index_store.refresh_counts(manifest.generation_id)
        except Exception as local_err:
            local_ok = False
            logger.error("local_index_write_failed", error_type=type(local_err).__name__)
        os_ok = os_remote_ok or local_ok
        qd_ok = qd_remote_ok or local_ok

        # 6. Record in PostgreSQL if session provided
        if db_session is not None:
            try:
                db_doc = IndexedDocument(
                    id=uuid.UUID(doc_uuid),
                    source_type=request.source_type,
                    source_id=request.source_id,
                    title=request.title,
                    url=request.url,
                    content_hash=content_hash,
                    extra_metadata=request.metadata,
                )
                await db_session.merge(db_doc)  # upsert by deterministic id
                await db_session.flush()
            except Exception as e:
                logger.warning("db_record_indexed_document_failed", error=str(e))

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "document_indexed_successfully",
            document_id=doc_uuid,
            title=request.title,
            chunks_count=len(chunks),
            elapsed_ms=elapsed_ms,
        )

        return IngestDocumentResponse(
            document_id=doc_uuid,
            source_type=request.source_type,
            source_id=request.source_id,
            total_chunks=len(chunks),
            indexed_in_opensearch=os_ok,
            indexed_in_qdrant=qd_ok,
            status="success" if (os_ok and qd_ok) else "partial",
            message=f"Indexed {len(chunks)} chunks in {elapsed_ms}ms",
        )

    async def _index_opensearch_remote(self, chunks: list[DocumentChunk]) -> bool:
        """Bulk-index into a remote OpenSearch if reachable. Returns False when unavailable."""
        if self._opensearch_alive is False:
            return False
        try:
            bulk_body: list[dict[str, Any]] = []
            for c in chunks:
                bulk_body.append({"index": {"_index": self.settings.opensearch_index_name, "_id": c.id}})
                bulk_body.append(
                    {
                        "chunk_id": c.id,
                        "document_id": c.document_id,
                        "source_type": c.source_type,
                        "source_id": c.source_id,
                        "title": c.title,
                        "url": c.url,
                        "content": c.content,
                        "chunk_index": c.chunk_index,
                        "total_chunks": c.total_chunks,
                        "start_char": c.start_char,
                        "end_char": c.end_char,
                        "metadata": c.metadata,
                    }
                )
            res = await self.opensearch_client.bulk(body=bulk_body, refresh=True)
            if not res.get("errors", False):
                self._opensearch_alive = True
                return True
        except Exception as e:
            self._opensearch_alive = False
            logger.info("opensearch_server_unavailable", error_type=type(e).__name__)
        return False

    async def _index_qdrant_remote(self, chunks: list[DocumentChunk], generation_id: str) -> bool:
        """Upsert into a remote Qdrant server if reachable. Returns False when unavailable."""
        if self._qdrant_alive is False:
            return False
        points = [
            qmodels.PointStruct(
                id=c.id,
                vector=c.embedding,
                payload={
                    "chunk_id": c.id,
                    "document_id": c.document_id,
                    "source_type": c.source_type,
                    "source_id": c.source_id,
                    "title": c.title,
                    "url": c.url,
                    "content": c.content,
                    "chunk_index": c.chunk_index,
                    "total_chunks": c.total_chunks,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "metadata": c.metadata,
                    "index_generation": generation_id,
                    "embedding_model": self.embedding_provider.model_name,
                },
            )
            for c in chunks
            if c.embedding is not None
        ]
        if not points:
            return False
        try:
            await self.qdrant_client.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=points,
                wait=True,
            )
            self._qdrant_alive = True
            return True
        except Exception as e:
            self._qdrant_alive = False
            logger.info("qdrant_server_unavailable", error_type=type(e).__name__)
        return False

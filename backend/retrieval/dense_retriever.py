"""
Dense / Vector Retriever backed by Qdrant (remote server if reachable, otherwise the
canonical embedded local store) and an EmbeddingProvider.

Fail-closed: before every search the active index manifest is validated against the
live embedding provider's identity. A missing, incomplete or incompatible index raises
an IndexIntegrityError; results computed from mismatched vectors are never returned.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from core.config import Settings, get_settings
from core.exceptions import AppException
from embedding.base import EmbeddingProvider
from embedding.factory import get_embedding_provider
from retrieval.base import BaseRetriever
from retrieval.index_manifest import IndexStore
from retrieval.models import RetrievalQuery, RetrievalResult, ScoredChunk

logger = structlog.get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Retrieves document passages by generating query embeddings and searching Qdrant.
    """

    def __init__(
        self,
        client: AsyncQdrantClient | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        collection_name: str | None = None,
        settings: Settings | None = None,
        index_store: IndexStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = collection_name or self.settings.qdrant_collection_name
        self._explicit_provider = embedding_provider
        self.index_store = index_store or IndexStore(settings=self.settings)
        self._server_available: bool | None = None

        self.client = client or AsyncQdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            api_key=self.settings.qdrant_api_key,
            timeout=2.0,  # Fast fail on local dev if cluster offline
        )

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        # Resolved lazily so constructing a retriever never loads a model.
        return self._explicit_provider or get_embedding_provider()

    @property
    def name(self) -> str:
        return "dense"

    async def aclose(self) -> None:
        """Close the Qdrant client (idempotent)."""
        try:
            await self.client.close()
        except Exception as e:
            logger.debug("qdrant_client_close_failed", error_type=type(e).__name__)

    async def ping(self) -> bool:
        """Check if Qdrant is alive and accessible."""
        try:
            await self.client.get_collections()
            self._server_available = True
            return True
        except Exception as e:
            self._server_available = False
            logger.warning("qdrant_ping_failed", error_type=type(e).__name__)
            return False

    def _build_filter(self, query: RetrievalQuery) -> qmodels.Filter | None:
        """Build Qdrant Filter from RetrievalQuery filters and source_types."""
        must_conditions: list[qmodels.Condition] = []

        if query.source_types:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="source_type",
                    match=qmodels.MatchAny(any=query.source_types),
                )
            )

        if query.filters:
            for k, v in query.filters.items():
                if isinstance(v, list):
                    must_conditions.append(
                        qmodels.FieldCondition(key=f"metadata.{k}", match=qmodels.MatchAny(any=v))
                    )
                else:
                    must_conditions.append(
                        qmodels.FieldCondition(key=f"metadata.{k}", match=qmodels.MatchValue(value=v))
                    )

        if not must_conditions:
            return None
        return qmodels.Filter(must=must_conditions)

    @staticmethod
    def _to_chunks(points: list[Any]) -> list[ScoredChunk]:
        chunks: list[ScoredChunk] = []
        for rank, pt in enumerate(points, start=1):
            payload = pt.payload or {}
            score = float(pt.score)
            chunks.append(
                ScoredChunk(
                    chunk_id=payload.get("chunk_id", str(pt.id)),
                    document_id=payload.get("document_id", ""),
                    source_type=payload.get("source_type", "local"),
                    source_id=payload.get("source_id", ""),
                    title=payload.get("title", ""),
                    url=payload.get("url"),
                    content=payload.get("content", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    total_chunks=payload.get("total_chunks", 1),
                    score=score,
                    retriever_name="dense",
                    rank=rank,
                    dense_score=score,
                    dense_rank=rank,
                    metadata=payload.get("metadata", {}),
                )
            )
        return chunks

    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        """Embed the query and run a similarity search against the validated index."""
        start_time = time.perf_counter()

        # Fail closed: manifest must exist, be complete and match the provider identity.
        provider = self.embedding_provider
        # validate_for_read() may load the embedding model (first call imports torch and
        # reads weights; tens of seconds when cold). Run it in a worker thread so the
        # event loop - and every other request/health check - is not frozen meanwhile.
        manifest = await asyncio.to_thread(self.index_store.validate_for_read, provider)
        diag = {
            "embedding_model": manifest.embedding.model_name,
            "index_generation": manifest.generation_id,
            "embedding_is_mock": manifest.embedding.is_mock,
        }

        try:
            query_vector = await provider.aembed_query(query.query_text)
            qfilter = self._build_filter(query)

            points: list[Any] | None = None
            source = ""

            # 1. Remote cluster, if not known to be offline
            if self._server_available is not False:
                try:
                    res = await self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        query_filter=qfilter,
                        limit=query.top_k,
                        with_payload=True,
                    )
                    self._server_available = True
                    points = list(res.points)
                    source = "qdrant_server"
                except Exception as remote_err:
                    self._server_available = False
                    logger.info(
                        "qdrant_remote_unavailable_using_local_storage",
                        error_type=type(remote_err).__name__,
                    )

            # 2. Canonical embedded local store
            if points is None:
                with self.index_store.qdrant(
                    manifest.generation_id, manifest.embedding.dimension
                ) as qc:
                    res = qc.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        query_filter=qfilter,
                        limit=query.top_k,
                        with_payload=True,
                    )
                points = list(res.points)
                source = "local_qdrant_storage"

            for pt in points:
                self.index_store.check_generation(
                    manifest, (pt.payload or {}).get("index_generation")
                )

            chunks = self._to_chunks(points)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return RetrievalResult(
                query_text=query.query_text,
                retriever="dense",
                chunks=chunks,
                total_found=len(chunks),
                latency_ms=latency_ms,
                metadata={"source": source, **diag},
            )

        except AppException:
            raise  # integrity / embedding errors are never swallowed into empty results
        except Exception as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error("dense_search_error", error_type=type(e).__name__, latency_ms=latency_ms)
            return RetrievalResult(
                query_text=query.query_text,
                retriever="dense",
                chunks=[],
                total_found=0,
                latency_ms=latency_ms,
                metadata={"error": type(e).__name__, **diag},
            )

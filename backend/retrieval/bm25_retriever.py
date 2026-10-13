"""
Lexical / BM25 Retriever backed by OpenSearch with LocalBM25Store fallback.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any
import structlog
from opensearchpy import AsyncOpenSearch

from core.config import Settings, get_settings
from retrieval.base import BaseRetriever
from retrieval.index_manifest import IndexStore
from retrieval.local_bm25 import LocalBM25Store
from retrieval.models import RetrievalQuery, RetrievalResult, ScoredChunk

logger = structlog.get_logger(__name__)


class BM25Retriever(BaseRetriever):
    """
    Retrieves document passages using OpenSearch BM25 keyword matching with field boosting.
    Falls back to LocalBM25Store for offline/local development without Docker.
    """

    def __init__(
        self,
        client: AsyncOpenSearch | None = None,
        index_name: str | None = None,
        local_store_path: Path | str | None = None,
        settings: Settings | None = None,
        index_store: IndexStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.index_name = index_name or self.settings.opensearch_index_name
        self.index_store = index_store or IndexStore(settings=self.settings)
        # Explicit path = legacy/experimental override (no manifest). Normal operation
        # resolves the active generation under the canonical data root.
        self._explicit_local_path = Path(local_store_path) if local_store_path else None
        self._server_available: bool | None = None

        self.client = client or AsyncOpenSearch(
            hosts=[{"host": self.settings.opensearch_host, "port": self.settings.opensearch_port}],
            http_auth=(
                (self.settings.opensearch_user, self.settings.opensearch_password)
                if self.settings.opensearch_user and self.settings.opensearch_password
                else None
            ),
            use_ssl=self.settings.opensearch_use_ssl,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=2.0,  # fast fail on local dev if cluster offline
        )

    @property
    def name(self) -> str:
        return "bm25"

    async def aclose(self) -> None:
        """Close the OpenSearch client session (idempotent)."""
        try:
            await self.client.close()
        except Exception as e:
            logger.debug("opensearch_client_close_failed", error_type=type(e).__name__)

    async def ping(self) -> bool:
        """Check if OpenSearch is alive and accessible."""
        try:
            res = bool(await self.client.ping())
            self._server_available = res
            return res
        except Exception as e:
            self._server_available = False
            logger.warning("opensearch_ping_failed", error=str(e))
            return False

    def build_query(self, query: RetrievalQuery) -> dict[str, Any]:
        """
        Construct OpenSearch BM25 search DSL with field boosting and optional filters.
        """
        must_clauses: list[dict[str, Any]] = [
            {
                "multi_match": {
                    "query": query.query_text,
                    "fields": [
                        "title^2.5",
                        "content^1.0",
                        "metadata.abstract^1.8",
                        "metadata.authors^1.2",
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "prefix_length": 2,
                }
            }
        ]

        filter_clauses: list[dict[str, Any]] = []

        if query.source_types:
            filter_clauses.append({"terms": {"source_type": query.source_types}})

        if query.filters:
            for k, v in query.filters.items():
                if isinstance(v, list):
                    filter_clauses.append({"terms": {f"metadata.{k}": v}})
                else:
                    filter_clauses.append({"term": {f"metadata.{k}": v}})

        bool_query: dict[str, Any] = {"must": must_clauses}
        if filter_clauses:
            bool_query["filter"] = filter_clauses

        return {
            "size": query.top_k,
            "query": {"bool": bool_query},
            "_source": [
                "chunk_id",
                "document_id",
                "source_type",
                "source_id",
                "title",
                "url",
                "content",
                "chunk_index",
                "total_chunks",
                "metadata",
            ],
        }

    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Execute BM25 search query against OpenSearch (or LocalBM25Store fallback).
        """
        start_time = time.perf_counter()

        if self._server_available is not False:
            try:
                query_dsl = self.build_query(query)
                response = await self.client.search(
                    index=self.index_name,
                    body=query_dsl,
                )
                self._server_available = True
                hits = response.get("hits", {}).get("hits", [])
                total_hits = response.get("hits", {}).get("total", {}).get("value", len(hits))

                scored_chunks: list[ScoredChunk] = []
                for rank, hit in enumerate(hits, start=1):
                    source = hit.get("_source", {})
                    score = float(hit.get("_score", 0.0))

                    scored_chunks.append(
                        ScoredChunk(
                            chunk_id=source.get("chunk_id", hit.get("_id", "")),
                            document_id=source.get("document_id", ""),
                            source_type=source.get("source_type", "local"),
                            source_id=source.get("source_id", ""),
                            title=source.get("title", ""),
                            url=source.get("url"),
                            content=source.get("content", ""),
                            chunk_index=source.get("chunk_index", 0),
                            total_chunks=source.get("total_chunks", 1),
                            score=score,
                            retriever_name="bm25",
                            rank=rank,
                            bm25_score=score,
                            bm25_rank=rank,
                            metadata=source.get("metadata", {}),
                        )
                    )

                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return RetrievalResult(
                    query_text=query.query_text,
                    retriever="bm25",
                    chunks=scored_chunks,
                    total_found=int(total_hits),
                    latency_ms=latency_ms,
                )

            except Exception as e:
                self._server_available = False
                logger.info("opensearch_cluster_unavailable_using_local_store", error=str(e))

        # Local fallback execution (canonical, generation-validated store)
        if self._explicit_local_path is not None:
            if not self._explicit_local_path.exists():
                return RetrievalResult(
                    query_text=query.query_text,
                    retriever="bm25",
                    chunks=[],
                    total_found=0,
                    latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
                    metadata={"error": "no_bm25_source_available"},
                )
            local_store = LocalBM25Store(storage_file=self._explicit_local_path)
            meta: dict[str, Any] = {"fallback": "local_bm25_store"}
        else:
            manifest = self.index_store.active_manifest()  # fail closed
            local_store = LocalBM25Store(
                storage_file=self.index_store.bm25_path(manifest.generation_id)
            )
            self.index_store.check_generation(manifest, local_store.index_generation)
            meta = {"fallback": "local_bm25_store", "index_generation": manifest.generation_id}

        local_chunks = local_store.search(query)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return RetrievalResult(
            query_text=query.query_text,
            retriever="bm25",
            chunks=local_chunks,
            total_found=len(local_chunks),
            latency_ms=latency_ms,
            metadata=meta,
        )

"""
Index manifest, integrity errors and the generation-based IndexStore.

Layout (see core/paths.py):

    <data_root>/indexes/ACTIVE                       -> {"generation_id": ...}
    <data_root>/indexes/<generation_id>/manifest.json
    <data_root>/indexes/<generation_id>/bm25_index.json
    <data_root>/indexes/<generation_id>/qdrant/

Guarantees:
- Every persisted vector index has a manifest recording the exact embedding
  identity (provider, model, dimension, revision, normalization), the chunking
  configuration and the document/chunk counts.
- Readers and writers validate model IDENTITY, not just dimension. A 384-dim mock
  vector is never accepted as an all-MiniLM-L6-v2 vector.
- Rebuilds go to a fresh generation directory and are activated atomically only
  after verification; an interrupted build leaves the previous generation active.
- Missing / incomplete / incompatible manifests FAIL CLOSED.
"""
from __future__ import annotations

import json
import math
import os
import secrets
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import structlog
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from core.config import Settings, get_settings
from core.exceptions import AppException
from core.paths import indexes_root, resolve_data_root
from embedding.identity import EmbeddingIdentity, describe_embedding_provider
from retrieval.local_bm25 import LocalBM25Store
from retrieval.models import DocumentChunk

logger = structlog.get_logger(__name__)

MANIFEST_SCHEMA_VERSION = 1
CHUNKER_VERSION = "passage_chunker/v1"
_ID_NAMESPACE = uuid.UUID("6f1d2c3e-5a7b-4c8d-9e0f-1a2b3c4d5e6f")


# ── Errors ────────────────────────────────────────────────────────────────────

class IndexIntegrityError(AppException):
    """Base class: the persisted index cannot be trusted. Messages are client-safe."""
    status_code = 503
    code = "index_integrity_error"


class ManifestMissingError(IndexIntegrityError):
    code = "index_manifest_missing"


class InterruptedBuildError(IndexIntegrityError):
    code = "index_build_incomplete"


class EmbeddingModelMismatchError(IndexIntegrityError):
    code = "embedding_model_mismatch"


class EmbeddingDimensionMismatchError(IndexIntegrityError):
    code = "embedding_dimension_mismatch"


class IndexGenerationMismatchError(IndexIntegrityError):
    code = "index_generation_mismatch"


class StaleIndexError(IndexIntegrityError):
    code = "index_stale"


# ── Manifest models ───────────────────────────────────────────────────────────

class ChunkingConfig(BaseModel):
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    chunker_version: str = CHUNKER_VERSION


class IndexCounts(BaseModel):
    documents: int = 0
    chunks: int = 0


class IndexManifest(BaseModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    generation_id: str
    status: str = "building"            # "building" | "complete"
    created_at: str
    completed_at: str | None = None
    embedding: EmbeddingIdentity
    chunking: ChunkingConfig
    counts: IndexCounts = Field(default_factory=IndexCounts)
    collection_name: str = "document_chunks"
    corpus: dict[str, Any] = Field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def deterministic_document_id(source_type: str, source_id: str) -> str:
    """Stable document UUID so re-indexing the same source replaces, never duplicates."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"doc:{source_type}:{source_id}"))


def deterministic_chunk_id(document_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"chunk:{document_id}:{chunk_index}"))


def new_generation_id() -> str:
    return f"gen-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate_vectors(
    vectors: list[list[float]],
    expected_dim: int,
    normalized: bool = True,
) -> None:
    """Reject wrong-dimension, non-finite, zero or (when configured) non-unit vectors."""
    for i, v in enumerate(vectors):
        if len(v) != expected_dim:
            raise EmbeddingDimensionMismatchError(
                f"Embedding {i} has dimension {len(v)}; index requires {expected_dim}."
            )
        if not all(math.isfinite(x) for x in v):
            raise IndexIntegrityError(f"Embedding {i} contains non-finite values.")
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0.0:
            raise IndexIntegrityError(f"Embedding {i} is the zero vector.")
        if normalized and abs(norm - 1.0) > 1e-3:
            raise IndexIntegrityError(
                f"Embedding {i} is not L2-normalized (norm={norm:.4f}) but the index manifest "
                "declares normalized embeddings."
            )


def validate_embedding_identity(manifest: IndexManifest, identity: EmbeddingIdentity) -> None:
    """Compare the live provider identity against the manifest. Raises on any mismatch."""
    m = manifest.embedding
    if m.is_mock != identity.is_mock:
        raise EmbeddingModelMismatchError(
            f"Index was built with a {'MOCK' if m.is_mock else 'real'} embedder "
            f"('{m.model_name}') but the active provider is "
            f"{'MOCK' if identity.is_mock else 'real'} ('{identity.model_name}'). "
            "Rebuild the index (scripts/index_admin.py rebuild)."
        )
    if m.provider != identity.provider or m.model_name != identity.model_name:
        raise EmbeddingModelMismatchError(
            f"Index embedding model is '{m.provider}:{m.model_name}' but the active provider is "
            f"'{identity.provider}:{identity.model_name}'. Rebuild the index."
        )
    if m.dimension != identity.dimension:
        raise EmbeddingDimensionMismatchError(
            f"Index dimension is {m.dimension} but the active provider produces "
            f"{identity.dimension}. Rebuild the index."
        )
    if m.normalized != identity.normalized:
        raise EmbeddingModelMismatchError(
            f"Index normalization={m.normalized} differs from provider normalization="
            f"{identity.normalized}. Rebuild the index."
        )
    if m.revision and identity.revision and m.revision != identity.revision:
        raise EmbeddingModelMismatchError(
            "Embedding model revision differs from the revision the index was built with "
            f"({m.revision[:12]} vs {identity.revision[:12]}). Rebuild the index."
        )


# ── IndexStore ────────────────────────────────────────────────────────────────

class IndexStore:
    """Resolves, validates and mutates index generations under the canonical data root."""

    def __init__(self, settings: Settings | None = None, data_root: Path | str | None = None) -> None:
        self.settings = settings or get_settings()
        self.data_root = Path(data_root).resolve() if data_root else resolve_data_root(self.settings)
        self.root = self.data_root / "indexes" if data_root else indexes_root(self.settings)
        self.collection_name = self.settings.qdrant_collection_name

    # -- paths --------------------------------------------------------------
    def gen_dir(self, gid: str) -> Path:
        return self.root / gid

    def manifest_path(self, gid: str) -> Path:
        return self.gen_dir(gid) / "manifest.json"

    def bm25_path(self, gid: str) -> Path:
        return self.gen_dir(gid) / "bm25_index.json"

    def qdrant_dir(self, gid: str) -> Path:
        return self.gen_dir(gid) / "qdrant"

    @property
    def active_pointer_path(self) -> Path:
        return self.root / "ACTIVE"

    # -- pointer / manifest -------------------------------------------------
    def active_generation_id(self) -> str | None:
        try:
            data = json.loads(self.active_pointer_path.read_text(encoding="utf-8"))
            gid = data.get("generation_id")
            return str(gid) if gid else None
        except FileNotFoundError:
            return None
        except Exception as e:
            raise ManifestMissingError(
                f"Active index pointer is unreadable ({type(e).__name__}). Rebuild the index."
            ) from e

    def read_manifest(self, gid: str) -> IndexManifest:
        path = self.manifest_path(gid)
        try:
            return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise ManifestMissingError(
                f"Index generation '{gid}' has no manifest. Rebuild the index."
            ) from e
        except Exception as e:
            raise ManifestMissingError(
                f"Index manifest for '{gid}' is unreadable ({type(e).__name__}). Rebuild the index."
            ) from e

    def write_manifest(self, manifest: IndexManifest) -> None:
        atomic_write_json(self.manifest_path(manifest.generation_id), manifest.model_dump(mode="json"))

    def active_manifest(self) -> IndexManifest:
        gid = self.active_generation_id()
        if gid is None:
            raise ManifestMissingError(
                "No active vector index exists. Build one with "
                "`python scripts/index_admin.py rebuild`."
            )
        manifest = self.read_manifest(gid)
        if manifest.status != "complete":
            raise InterruptedBuildError(
                f"Index generation '{gid}' is not complete (status={manifest.status}). "
                "Rebuild the index."
            )
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            raise StaleIndexError(
                f"Index manifest schema v{manifest.schema_version} is not supported "
                f"(expected v{MANIFEST_SCHEMA_VERSION}). Rebuild the index."
            )
        if manifest.generation_id != gid:
            raise IndexGenerationMismatchError(
                "Manifest generation id does not match the active pointer. Rebuild the index."
            )
        return manifest

    # -- validation ---------------------------------------------------------
    def validate_for_read(self, provider: Any) -> IndexManifest:
        """Active manifest must exist, be complete and match the provider identity."""
        manifest = self.active_manifest()
        validate_embedding_identity(manifest, describe_embedding_provider(provider))
        return manifest

    def validate_for_write(self, provider: Any, chunking: ChunkingConfig) -> IndexManifest:
        manifest = self.validate_for_read(provider)
        c = manifest.chunking
        if (
            c.chunk_size_tokens != chunking.chunk_size_tokens
            or c.chunk_overlap_tokens != chunking.chunk_overlap_tokens
            or c.chunker_version != chunking.chunker_version
        ):
            raise StaleIndexError(
                "Chunking configuration differs from the one recorded in the index manifest "
                f"(index: {c.chunk_size_tokens}/{c.chunk_overlap_tokens} {c.chunker_version}; "
                f"current: {chunking.chunk_size_tokens}/{chunking.chunk_overlap_tokens} "
                f"{chunking.chunker_version}). Rebuild the index."
            )
        return manifest

    def check_generation(self, manifest: IndexManifest, observed: str | None) -> None:
        if observed != manifest.generation_id:
            raise IndexGenerationMismatchError(
                "Retrieved data belongs to a different index generation than the active "
                f"manifest ({observed!r} vs {manifest.generation_id!r}). Rebuild the index."
            )

    # -- generation lifecycle ----------------------------------------------
    def create_empty_generation(
        self,
        identity: EmbeddingIdentity,
        chunking: ChunkingConfig,
        corpus: dict[str, Any] | None = None,
        generation_id: str | None = None,
    ) -> IndexManifest:
        gid = generation_id or new_generation_id()
        manifest = IndexManifest(
            generation_id=gid,
            status="building",
            created_at=now_iso(),
            embedding=identity,
            chunking=chunking,
            collection_name=self.collection_name,
            corpus=corpus or {},
        )
        self.gen_dir(gid).mkdir(parents=True, exist_ok=False)
        self.write_manifest(manifest)
        return manifest

    def finalize_generation(self, gid: str) -> IndexManifest:
        manifest = self.read_manifest(gid)
        docs, chunks = self._bm25_counts(gid)
        manifest.counts = IndexCounts(documents=docs, chunks=chunks)
        manifest.status = "complete"
        manifest.completed_at = now_iso()
        self.write_manifest(manifest)
        return manifest

    def activate(self, gid: str) -> None:
        manifest = self.read_manifest(gid)
        if manifest.status != "complete":
            raise InterruptedBuildError(f"Refusing to activate incomplete generation '{gid}'.")
        atomic_write_json(
            self.active_pointer_path,
            {"generation_id": gid, "activated_at": now_iso()},
        )

    def ensure_active_for_write(self, provider: Any, chunking: ChunkingConfig) -> IndexManifest:
        """
        Validate the active generation for writing. Only when NO index exists at all
        (fresh install) is an empty generation created from the provider's identity.
        An existing-but-incompatible index is never silently replaced.
        """
        if self.active_generation_id() is None:
            manifest = self.create_empty_generation(describe_embedding_provider(provider), chunking)
            self.finalize_generation(manifest.generation_id)
            self.activate(manifest.generation_id)
            logger.info("index_generation_created", generation_id=manifest.generation_id)
        return self.validate_for_write(provider, chunking)

    def list_generations(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and p.name.startswith("gen-"))

    # -- data operations ----------------------------------------------------
    def _bm25_counts(self, gid: str) -> tuple[int, int]:
        path = self.bm25_path(gid)
        if not path.exists():
            return 0, 0
        store = LocalBM25Store(storage_file=path)
        return len({c.document_id for c in store.chunks.values()}), len(store.chunks)

    @contextmanager
    def qdrant(self, gid: str, dimension: int) -> Iterator[QdrantClient]:
        """Open the generation's embedded Qdrant (sequential, per-operation)."""
        qdir = self.qdrant_dir(gid)
        qdir.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(qdir))
        try:
            existing = {c.name for c in client.get_collections().collections}
            if self.collection_name not in existing:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(size=dimension, distance=qmodels.Distance.COSINE),
                )
            yield client
        finally:
            client.close()

    def upsert_document(
        self,
        gid: str,
        document_id: str,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Replace all chunks of `document_id` in both BM25 and Qdrant (duplicate-safe).
        Chunks must carry validated embeddings.
        """
        manifest = self.read_manifest(gid)
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
                    "index_generation": gid,
                    "embedding_model": manifest.embedding.model_name,
                },
            )
            for c in chunks
            if c.embedding is not None
        ]

        with self.qdrant(gid, manifest.embedding.dimension) as qc:
            qc.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id", match=qmodels.MatchValue(value=document_id)
                            )
                        ]
                    )
                ),
                wait=True,
            )
            if points:
                qc.upsert(collection_name=self.collection_name, points=points, wait=True)

        bm25 = LocalBM25Store(storage_file=self.bm25_path(gid))
        bm25.index_generation = gid
        bm25.remove_document(document_id, save=False)
        bm25.add_chunks(chunks)  # saves

    def refresh_counts(self, gid: str) -> IndexManifest:
        manifest = self.read_manifest(gid)
        docs, chunks = self._bm25_counts(gid)
        manifest.counts = IndexCounts(documents=docs, chunks=chunks)
        self.write_manifest(manifest)
        return manifest

    def verify_integrity(self, gid: str | None = None) -> dict[str, Any]:
        """
        Deep check: BM25 ids == Qdrant ids == manifest counts; every vector has the
        manifest dimension, is finite, and carries the right generation/model tag.
        """
        gid = gid or self.active_generation_id()
        errors: list[str] = []
        if gid is None:
            return {"ok": False, "generation_id": None, "errors": ["no active generation"]}
        manifest = self.read_manifest(gid)
        bm25 = LocalBM25Store(storage_file=self.bm25_path(gid))
        bm25_ids = set(bm25.chunks)
        if bm25.index_generation != gid:
            errors.append(f"BM25 store generation tag is {bm25.index_generation!r}")

        q_ids: set[str] = set()
        bad_vec = bad_gen = bad_model = 0
        with self.qdrant(gid, manifest.embedding.dimension) as qc:
            offset = None
            while True:
                points, offset = qc.scroll(
                    collection_name=self.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                for p in points:
                    q_ids.add(str(p.id))
                    vec = p.vector if isinstance(p.vector, list) else []
                    try:
                        validate_vectors([vec], manifest.embedding.dimension, manifest.embedding.normalized)
                    except IndexIntegrityError:
                        bad_vec += 1
                    pl = p.payload or {}
                    if pl.get("index_generation") != gid:
                        bad_gen += 1
                    if pl.get("embedding_model") != manifest.embedding.model_name:
                        bad_model += 1
                if offset is None:
                    break

        if bm25_ids != q_ids:
            errors.append(
                f"BM25/Qdrant chunk id mismatch (bm25_only={len(bm25_ids - q_ids)}, "
                f"qdrant_only={len(q_ids - bm25_ids)})"
            )
        if bad_vec:
            errors.append(f"{bad_vec} vectors failed dimension/finite/normalization checks")
        if bad_gen:
            errors.append(f"{bad_gen} vectors carry a different index generation")
        if bad_model:
            errors.append(f"{bad_model} vectors carry a different embedding model tag")
        docs = len({c.document_id for c in bm25.chunks.values()})
        if manifest.status == "complete" and (
            manifest.counts.chunks != len(bm25_ids) or manifest.counts.documents != docs
        ):
            errors.append(
                f"manifest counts ({manifest.counts.documents} docs/{manifest.counts.chunks} chunks) "
                f"differ from data ({docs}/{len(bm25_ids)})"
            )
        return {
            "ok": not errors,
            "generation_id": gid,
            "documents": docs,
            "chunks": len(bm25_ids),
            "qdrant_points": len(q_ids),
            "errors": errors,
        }

    # -- diagnostics (no filesystem paths, no secrets) ------------------------
    def diagnostics(self, provider: Any | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {"index_present": False, "compatible": False}
        try:
            manifest = self.active_manifest()
        except IndexIntegrityError as e:
            info["reason"] = e.code
            info["detail"] = e.message
            return info
        info.update(
            index_present=True,
            generation_id=manifest.generation_id,
            status=manifest.status,
            created_at=manifest.created_at,
            completed_at=manifest.completed_at,
            embedding={
                "provider": manifest.embedding.provider,
                "model": manifest.embedding.model_name,
                "dimension": manifest.embedding.dimension,
                "revision": manifest.embedding.revision,
                "normalized": manifest.embedding.normalized,
                "is_mock": manifest.embedding.is_mock,
            },
            chunking=manifest.chunking.model_dump(),
            counts=manifest.counts.model_dump(),
        )
        if provider is not None:
            try:
                validate_embedding_identity(manifest, describe_embedding_provider(provider))
                info["compatible"] = True
            except AppException as e:
                info["reason"] = e.code
                info["detail"] = e.message
        return info


def get_index_store(settings: Settings | None = None) -> IndexStore:
    return IndexStore(settings=settings)

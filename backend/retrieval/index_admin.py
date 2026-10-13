"""
Index administration: legacy inventory, backup, verified rebuild, cleanup.

Used by scripts/index_admin.py and by tests. Everything here is synchronous and
side-effect-explicit; no function silently falls back to mock embeddings.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import structlog

from core.config import Settings, get_settings
from embedding.identity import describe_embedding_provider
from retrieval.index_manifest import (
    CHUNKER_VERSION,
    ChunkingConfig,
    IndexIntegrityError,
    IndexStore,
    deterministic_chunk_id,
    deterministic_document_id,
    validate_vectors,
)
from retrieval.models import DocumentChunk
from retrieval.passage_chunker import PassageChunker

logger = structlog.get_logger(__name__)


# ── Legacy inventory & backup ─────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inventory_legacy_store(data_dir: Path) -> dict[str, Any]:
    """
    Inventory a pre-manifest store laid out as <data_dir>/opensearch_local/bm25_index.json
    and <data_dir>/qdrant_local/. Read-only.
    """
    out: dict[str, Any] = {"data_dir": str(data_dir), "exists": data_dir.exists()}
    bm25 = data_dir / "opensearch_local" / "bm25_index.json"
    out["bm25_file"] = bm25.exists()
    if bm25.exists():
        raw = json.loads(bm25.read_text(encoding="utf-8"))
        chunks = raw.get("chunks", [])
        by_doc: dict[str, list[dict[str, Any]]] = {}
        for c in chunks:
            by_doc.setdefault(f"{c.get('source_type')}:{c.get('source_id')}", []).append(c)
        out["chunks"] = len(chunks)
        out["distinct_documents"] = len(by_doc)
        out["distinct_chunk_texts"] = len({c.get("content") for c in chunks})
        out["embedding_dims"] = sorted({len(c["embedding"]) for c in chunks if c.get("embedding")})
        out["manifest_present"] = False
        out["documents"] = {
            k: {
                "chunks": len(v),
                "unique_texts": len({c.get("content") for c in v}),
                "title": (v[0].get("title") or "")[:70],
            }
            for k, v in by_doc.items()
        }
    meta = data_dir / "qdrant_local" / "meta.json"
    if meta.exists():
        collections = json.loads(meta.read_text(encoding="utf-8")).get("collections", {})
        out["qdrant_collections"] = {
            name: {"vector_size": (c.get("vectors") or {}).get("size")}
            for name, c in collections.items()
        }
    return out


def backup_paths(paths: list[Path], dest_root: Path, label: str = "pre_phase86") -> dict[str, Any]:
    """Copy directories/files into dest_root/<timestamp>_<label>/ with a sha256 manifest."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = dest_root / f"{stamp}_{label}"
    dest.mkdir(parents=True, exist_ok=False)
    files: list[dict[str, Any]] = []
    for src in paths:
        if not src.exists():
            continue
        target = dest / src.name if src.is_file() else dest / src.name
        if src.is_dir():
            shutil.copytree(src, target, ignore=shutil.ignore_patterns(".lock"))
        else:
            shutil.copy2(src, target)
    for f in sorted(p for p in dest.rglob("*") if p.is_file()):
        files.append(
            {"path": str(f.relative_to(dest)).replace("\\", "/"), "bytes": f.stat().st_size, "sha256": _sha256(f)}
        )
    manifest = {"created_at": datetime.now(UTC).isoformat(), "label": label, "files": files}
    (dest / "BACKUP_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"backup_dir": dest, "file_count": len(files)}


# ── Corpus ────────────────────────────────────────────────────────────────────

def load_corpus(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    docs = data["documents"] if isinstance(data, dict) else data
    return docs, hashlib.sha256(raw).hexdigest()


# ── Provider validation ───────────────────────────────────────────────────────

def validate_provider_sample(provider: Any, allow_mock: bool = False) -> dict[str, Any]:
    """
    Embed a small sample and sanity-check the provider before any index is built.
    Raises on mock providers unless allow_mock, on wrong dimension, non-finite or
    non-deterministic output, and (for real providers) on lack of semantic signal.
    """
    identity = describe_embedding_provider(provider)  # loads the model; raises on failure
    if identity.is_mock and not allow_mock:
        raise IndexIntegrityError(
            "Refusing to build an index with the MOCK embedder. Configure the real provider "
            "(EMBEDDING_PROVIDER=local) or pass an explicit mock opt-in for experiments."
        )
    sample = [
        "retrieval augmented generation combines a retriever with a generator",
        "a simple recipe for baking chocolate cake with flour and sugar",
        "dense passage retrieval encodes questions and passages for open-domain question answering",
    ]
    v1 = provider.embed_texts(sample)
    v2 = provider.embed_texts(sample)
    validate_vectors(v1, identity.dimension, identity.normalized)
    deterministic = all(
        math.isclose(a, b, abs_tol=1e-5) for x, y in zip(v1, v2) for a, b in zip(x, y)
    )
    if not deterministic:
        raise IndexIntegrityError("Embedding provider is non-deterministic on identical input.")

    def cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    related, unrelated = cos(v1[0], v1[2]), cos(v1[0], v1[1])
    if not identity.is_mock and not related > unrelated:
        raise IndexIntegrityError(
            "Embedding sanity check failed: related texts are not closer than unrelated ones."
        )
    return {
        "identity": identity.model_dump(),
        "dimension_ok": True,
        "deterministic": True,
        "related_cosine": round(related, 4),
        "unrelated_cosine": round(unrelated, 4),
    }


# ── Rebuild ───────────────────────────────────────────────────────────────────

def rebuild_index(
    store: IndexStore,
    provider: Any,
    corpus_docs: list[dict[str, Any]],
    corpus_sha256: str | None = None,
    settings: Settings | None = None,
    allow_mock: bool = False,
    activate: bool = True,
    progress: Callable[[str], None] | None = None,
    fail_after_documents: int | None = None,  # test hook: simulate an interruption
) -> dict[str, Any]:
    """
    Build a NEW index generation from authoritative source documents, verify it, and
    only then activate it. The previously active generation is never touched; if the
    build is interrupted or verification fails it stays active and the partial
    generation is left flagged as 'building'.
    """
    st = settings or get_settings()
    say = progress or (lambda m: None)
    sample = validate_provider_sample(provider, allow_mock=allow_mock)
    identity = describe_embedding_provider(provider)
    chunking = ChunkingConfig(
        chunk_size_tokens=st.chunk_size_tokens,
        chunk_overlap_tokens=st.chunk_overlap_tokens,
        chunker_version=CHUNKER_VERSION,
    )
    chunker = PassageChunker(
        chunk_size_tokens=chunking.chunk_size_tokens,
        chunk_overlap_tokens=chunking.chunk_overlap_tokens,
    )
    previous = store.active_generation_id()
    manifest = store.create_empty_generation(
        identity,
        chunking,
        corpus={
            "document_count_declared": len(corpus_docs),
            "sha256": corpus_sha256,
        },
    )
    gid = manifest.generation_id
    say(f"building generation {gid} with {identity.provider}:{identity.model_name} ({identity.dimension}d)")

    report: dict[str, Any] = {
        "generation_id": gid,
        "previous_generation_id": previous,
        "embedding": identity.model_dump(),
        "provider_sample": sample,
        "documents_in_corpus": len(corpus_docs),
        "documents_indexed": 0,
        "documents_skipped": [],
        "chunks_indexed": 0,
        "per_document": [],
    }

    seen: set[str] = set()
    try:
        for n, doc in enumerate(corpus_docs, start=1):
            if fail_after_documents is not None and n > fail_after_documents:
                raise RuntimeError("simulated interruption")
            stype = doc.get("source_type", "local")
            sid = str(doc.get("source_id", ""))
            key = f"{stype}:{sid}"
            content = (doc.get("content") or "").strip()
            if not sid or not content:
                report["documents_skipped"].append({"doc": key, "reason": "empty"})
                continue
            if key in seen:
                report["documents_skipped"].append({"doc": key, "reason": "duplicate_in_corpus"})
                continue
            seen.add(key)

            doc_id = deterministic_document_id(stype, sid)
            chunks: list[DocumentChunk] = chunker.chunk_document(
                document_id=doc_id,
                content=doc["content"],
                source_type=stype,
                source_id=sid,
                title=doc.get("title", ""),
                url=doc.get("url"),
                metadata=doc.get("metadata", {}),
            )
            for c in chunks:
                c.id = deterministic_chunk_id(doc_id, c.chunk_index)
            vectors = provider.embed_texts([c.content for c in chunks])
            validate_vectors(vectors, identity.dimension, identity.normalized)
            for c, v in zip(chunks, vectors):
                c.embedding = v
            store.upsert_document(gid, doc_id, chunks)
            report["documents_indexed"] += 1
            report["chunks_indexed"] += len(chunks)
            report["per_document"].append({"doc": key, "chunks": len(chunks)})
            say(f"  indexed {key}: {len(chunks)} chunks")

        final = store.finalize_generation(gid)
        verification = store.verify_integrity(gid)
        report["verification"] = verification
        if not verification["ok"]:
            raise IndexIntegrityError("Post-build verification failed: " + "; ".join(verification["errors"]))
        if final.counts.chunks != report["chunks_indexed"]:
            raise IndexIntegrityError("Chunk count mismatch between build report and manifest.")
        if activate:
            store.activate(gid)
        report["activated"] = activate
        report["status"] = "complete"
    except Exception as e:  # noqa: BLE001 - any failure leaves the old generation active
        report["status"] = "failed"
        report["error"] = f"{type(e).__name__}: {e}"
        report["activated"] = False
        report["active_generation_unchanged"] = store.active_generation_id() == previous
        logger.error("index_rebuild_failed", generation_id=gid, error_type=type(e).__name__)
    return report


# ── Cleanup / recovery ────────────────────────────────────────────────────────

def list_generation_status(store: IndexStore) -> list[dict[str, Any]]:
    active = store.active_generation_id()
    out: list[dict[str, Any]] = []
    for gid in store.list_generations():
        try:
            m = store.read_manifest(gid)
            out.append(
                {
                    "generation_id": gid,
                    "active": gid == active,
                    "status": m.status,
                    "model": m.embedding.model_name,
                    "mock": m.embedding.is_mock,
                    "documents": m.counts.documents,
                    "chunks": m.counts.chunks,
                    "created_at": m.created_at,
                }
            )
        except IndexIntegrityError:
            out.append({"generation_id": gid, "active": gid == active, "status": "no_manifest"})
    return out


def prune_incomplete(store: IndexStore) -> list[str]:
    """Delete generation directories that never completed. Never touches the active one."""
    removed: list[str] = []
    active = store.active_generation_id()
    for row in list_generation_status(store):
        if row["generation_id"] != active and row["status"] != "complete":
            shutil.rmtree(store.gen_dir(row["generation_id"]), ignore_errors=True)
            removed.append(row["generation_id"])
    return removed


def count_by_source(counter_docs: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(d.get("source_type", "local") for d in counter_docs))

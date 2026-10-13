"""
Phase 8.6 regression tests: index manifest, identity validation, fail-closed
behavior, canonical paths, duplicate-safe and interruption-safe rebuilds.

All tests are offline: embeddings come from explicit mock providers.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.config import Settings, get_settings
from core.paths import REPO_ROOT, resolve_data_root
from embedding.errors import EmbeddingModelLoadError
from embedding.factory import create_embedding_provider
from embedding.identity import EmbeddingIdentity
from embedding.mock_provider import MockEmbeddingProvider
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.index_admin import (
    list_generation_status,
    prune_incomplete,
    rebuild_index,
    validate_provider_sample,
)
from retrieval.index_manifest import (
    ChunkingConfig,
    EmbeddingDimensionMismatchError,
    EmbeddingModelMismatchError,
    IndexGenerationMismatchError,
    IndexIntegrityError,
    IndexStore,
    InterruptedBuildError,
    ManifestMissingError,
    StaleIndexError,
    deterministic_document_id,
    validate_vectors,
)
from retrieval.indexing_service import IndexingService
from retrieval.models import IngestDocumentRequest, RetrievalQuery

from tests.conftest import MOCK_CORPUS


class FakeRealProvider(MockEmbeddingProvider):
    """Stands in for 'a different model' without being flagged as mock by isinstance."""


class OtherModelProvider:
    """Non-mock provider with a configurable identity; deterministic unit vectors."""

    def __init__(self, model_name="all-MiniLM-L6-v2", dimension=384, revision=None):
        self._m, self._d, self.revision = model_name, dimension, revision
        self.normalize_embeddings = True

    model_name = property(lambda self: self._m)
    dimension = property(lambda self: self._d)

    def embed_texts(self, texts):
        """Deterministic, L2-normalized bag-of-words hashing (semantically non-trivial)."""
        import math
        import re
        import zlib

        out = []
        for t in texts:
            v = [0.0] * self._d
            for w in re.findall(r"\w+", t.lower()):
                if w in {"a", "an", "the", "with", "and", "for", "of", "to"}:
                    continue
                v[zlib.crc32(w.encode()) % self._d] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    def embed_query(self, text):
        return self.embed_texts([text])[0]

    async def aembed_texts(self, texts):
        return self.embed_texts(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


def _store(tmp_path: Path) -> IndexStore:
    return IndexStore(settings=get_settings(), data_root=tmp_path / "idx")


def _mock_build(store: IndexStore, provider=None, docs=None, **kw):
    return rebuild_index(
        store,
        provider or MockEmbeddingProvider(dimension=384),
        docs or MOCK_CORPUS,
        settings=get_settings(),
        allow_mock=True,
        **kw,
    )


# ── Manifest content ──────────────────────────────────────────────────────────

def test_manifest_records_required_fields(tmp_path):
    store = _store(tmp_path)
    rep = _mock_build(store)
    assert rep["status"] == "complete"
    m = store.active_manifest()
    assert m.status == "complete"
    assert m.embedding.provider == "mock" and m.embedding.is_mock
    assert m.embedding.model_name == "mock-minilm-384"
    assert m.embedding.dimension == 384 and m.embedding.normalized
    assert m.chunking.chunk_size_tokens == get_settings().chunk_size_tokens
    assert m.chunking.chunker_version
    assert m.counts.documents == 3 and m.counts.chunks >= 3
    assert m.generation_id.startswith("gen-") and m.created_at and m.completed_at


# ── Identity / dimension / missing manifest ──────────────────────────────────

def test_missing_manifest_fails_closed(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ManifestMissingError):
        store.active_manifest()
    with pytest.raises(ManifestMissingError):
        store.validate_for_read(MockEmbeddingProvider(dimension=384))


@pytest.mark.asyncio
async def test_dense_retriever_fails_closed_without_manifest(tmp_path):
    r = DenseRetriever(embedding_provider=MockEmbeddingProvider(dimension=384),
                       index_store=_store(tmp_path))
    with pytest.raises(ManifestMissingError):
        await r.search(RetrievalQuery(query_text="anything"))


@pytest.mark.asyncio
async def test_bm25_fails_closed_without_manifest(tmp_path):
    r = BM25Retriever(index_store=_store(tmp_path))
    r._server_available = False
    with pytest.raises(ManifestMissingError):
        await r.search(RetrievalQuery(query_text="anything"))


def test_mock_vectors_never_accepted_as_real_model(tmp_path):
    """Same dimension (384), different identity: must be rejected."""
    store = _store(tmp_path)
    _mock_build(store)  # index of mock 384-d vectors
    real_like = OtherModelProvider("all-MiniLM-L6-v2", 384)
    with pytest.raises(EmbeddingModelMismatchError, match="MOCK"):
        store.validate_for_read(real_like)


def test_model_identity_mismatch(tmp_path):
    store = _store(tmp_path)
    store_prov = OtherModelProvider("all-MiniLM-L6-v2", 384)
    rep = rebuild_index(store, store_prov, MOCK_CORPUS, settings=get_settings())
    assert rep["status"] == "complete", rep
    with pytest.raises(EmbeddingModelMismatchError):
        store.validate_for_read(OtherModelProvider("paraphrase-MiniLM-L3-v2", 384))


def test_dimension_mismatch(tmp_path):
    store = _store(tmp_path)
    rebuild_index(store, OtherModelProvider("m", 384), MOCK_CORPUS, settings=get_settings())
    with pytest.raises(EmbeddingDimensionMismatchError):
        store.validate_for_read(OtherModelProvider("m", 768))


def test_revision_mismatch(tmp_path):
    store = _store(tmp_path)
    rebuild_index(store, OtherModelProvider("m", 384, revision="aaa111"), MOCK_CORPUS, settings=get_settings())
    with pytest.raises(EmbeddingModelMismatchError, match="revision"):
        store.validate_for_read(OtherModelProvider("m", 384, revision="bbb222"))


def test_validate_vectors_rejects_bad_vectors():
    with pytest.raises(EmbeddingDimensionMismatchError):
        validate_vectors([[1.0, 0.0]], 3)
    with pytest.raises(IndexIntegrityError):
        validate_vectors([[0.0, 0.0, 0.0]], 3)
    with pytest.raises(IndexIntegrityError):
        validate_vectors([[float("nan"), 0.0, 0.0]], 3)
    with pytest.raises(IndexIntegrityError):
        validate_vectors([[2.0, 0.0, 0.0]], 3, normalized=True)
    validate_vectors([[1.0, 0.0, 0.0]], 3)


# ── Mock provider is opt-in only ──────────────────────────────────────────────

def test_mock_not_silently_selected_when_real_model_fails(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "definitely/not-a-real-model-xyz")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    get_settings.cache_clear()
    provider = create_embedding_provider(get_settings())
    assert not isinstance(provider, MockEmbeddingProvider)
    with pytest.raises(EmbeddingModelLoadError):
        provider.embed_query("hello")
    with pytest.raises(EmbeddingModelLoadError):
        _ = provider.dimension


def test_mock_selected_only_when_explicitly_configured(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    get_settings.cache_clear()
    assert isinstance(create_embedding_provider(get_settings()), MockEmbeddingProvider)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    get_settings.cache_clear()
    assert not isinstance(create_embedding_provider(get_settings()), MockEmbeddingProvider)


def test_rebuild_refuses_mock_without_opt_in(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(IndexIntegrityError, match="MOCK"):
        rebuild_index(store, MockEmbeddingProvider(dimension=384), MOCK_CORPUS,
                      settings=get_settings(), allow_mock=False)
    with pytest.raises(IndexIntegrityError, match="MOCK"):
        validate_provider_sample(MockEmbeddingProvider(dimension=384), allow_mock=False)
    assert store.active_generation_id() is None
    assert store.list_generations() == []  # nothing was written


# ── Generation mismatch / stale / interruption ───────────────────────────────

def test_index_generation_mismatch_detected(tmp_path):
    store = _store(tmp_path)
    _mock_build(store)
    m = store.active_manifest()
    with pytest.raises(IndexGenerationMismatchError):
        store.check_generation(m, "gen-some-other-generation")
    with pytest.raises(IndexGenerationMismatchError):
        store.check_generation(m, None)


@pytest.mark.asyncio
async def test_bm25_detects_foreign_generation_tag(tmp_path):
    from retrieval.local_bm25 import LocalBM25Store

    store = _store(tmp_path)
    _mock_build(store)
    gid = store.active_generation_id()
    bm = LocalBM25Store(storage_file=store.bm25_path(gid))
    bm.index_generation = "gen-tampered"
    bm._save()
    r = BM25Retriever(index_store=store)
    r._server_available = False
    with pytest.raises(IndexGenerationMismatchError):
        await r.search(RetrievalQuery(query_text="retrieval"))


def test_stale_chunking_config_rejected_for_writes(tmp_path):
    store = _store(tmp_path)
    prov = MockEmbeddingProvider(dimension=384)
    _mock_build(store, prov)
    other = ChunkingConfig(chunk_size_tokens=64, chunk_overlap_tokens=8, chunker_version="passage_chunker/v1")
    with pytest.raises(StaleIndexError):
        store.validate_for_write(prov, other)


def test_stale_schema_version_rejected(tmp_path):
    store = _store(tmp_path)
    _mock_build(store)
    m = store.active_manifest()
    m.schema_version = 0
    store.write_manifest(m)
    with pytest.raises(StaleIndexError):
        store.active_manifest()


def test_interrupted_rebuild_keeps_previous_generation_active(tmp_path):
    store = _store(tmp_path)
    good = _mock_build(store)
    good_gid = good["generation_id"]
    bad = _mock_build(store, fail_after_documents=1)
    assert bad["status"] == "failed"
    assert bad["active_generation_unchanged"] is True
    assert store.active_generation_id() == good_gid
    # Partial generation exists but is flagged and never served.
    rows = {r["generation_id"]: r for r in list_generation_status(store)}
    assert rows[bad["generation_id"]]["status"] == "building"
    with pytest.raises(InterruptedBuildError):
        store.activate(bad["generation_id"])
    # Previous index still fully verifiable and queryable.
    assert store.verify_integrity()["ok"]
    # Recovery: prune partial generation, active one untouched.
    assert prune_incomplete(store) == [bad["generation_id"]]
    assert store.active_generation_id() == good_gid and store.verify_integrity()["ok"]


def test_incomplete_active_generation_is_rejected(tmp_path):
    store = _store(tmp_path)
    _mock_build(store)
    m = store.active_manifest()
    m.status = "building"
    store.write_manifest(m)
    with pytest.raises(InterruptedBuildError):
        store.validate_for_read(MockEmbeddingProvider(dimension=384))


# ── Duplicate-safe re-indexing ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reindexing_same_document_does_not_duplicate(tmp_path, mock_embeddings_configured):
    svc = IndexingService(index_store=_store(tmp_path),
                          embedding_provider=MockEmbeddingProvider(dimension=384))
    svc._opensearch_alive = False
    svc._qdrant_alive = False
    req = IngestDocumentRequest(source_type="arxiv", source_id="X.1", title="T",
                                content="alpha beta gamma delta " * 20)
    r1 = await svc.index_document(req)
    r2 = await svc.index_document(req)
    r3 = await svc.index_document(req)
    assert r1.document_id == r2.document_id == r3.document_id == deterministic_document_id("arxiv", "X.1")
    store = svc.index_store
    rep = store.verify_integrity()
    assert rep["ok"], rep
    assert rep["chunks"] == r1.total_chunks
    assert rep["documents"] == 1
    assert store.active_manifest().counts.chunks == r1.total_chunks


@pytest.mark.asyncio
async def test_changed_document_replaces_old_chunks(tmp_path):
    svc = IndexingService(index_store=_store(tmp_path),
                          embedding_provider=MockEmbeddingProvider(dimension=384))
    svc._opensearch_alive = False
    svc._qdrant_alive = False
    long = IngestDocumentRequest(source_type="web", source_id="u", title="T", content="word " * 3000)
    short = IngestDocumentRequest(source_type="web", source_id="u", title="T", content="short text here")
    await svc.index_document(long)
    n_long = svc.index_store.active_manifest().counts.chunks
    assert n_long > 1
    await svc.index_document(short)
    rep = svc.index_store.verify_integrity()
    assert rep["ok"] and rep["chunks"] == 1


def test_rebuild_dedupes_corpus_and_reports_skips(tmp_path):
    store = _store(tmp_path)
    docs = MOCK_CORPUS + [dict(MOCK_CORPUS[0]), {"source_type": "local", "source_id": "e", "content": "  "}]
    rep = _mock_build(store, docs=docs)
    assert rep["documents_in_corpus"] == 5
    assert rep["documents_indexed"] == 3
    reasons = sorted(s["reason"] for s in rep["documents_skipped"])
    assert reasons == ["duplicate_in_corpus", "empty"]


def test_indexing_into_incompatible_index_is_refused(tmp_path):
    """A writer with a different model must not append vectors to an existing index."""
    store = _store(tmp_path)
    _mock_build(store)
    svc = IndexingService(index_store=store, embedding_provider=OtherModelProvider("all-MiniLM-L6-v2", 384))
    with pytest.raises(EmbeddingModelMismatchError):
        import asyncio
        asyncio.run(svc.index_document(
            IngestDocumentRequest(source_type="local", source_id="n", title="t", content="hello world")))
    assert store.verify_integrity()["ok"]


# ── Canonical paths ───────────────────────────────────────────────────────────

def test_default_data_root_is_repo_data_not_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("DATA_DIR", raising=False)
    get_settings.cache_clear()
    monkeypatch.chdir(tmp_path)
    assert resolve_data_root(Settings(postgres_password="x")) == REPO_ROOT / "data"
    monkeypatch.chdir(REPO_ROOT / "backend")
    assert resolve_data_root(Settings(postgres_password="x")) == REPO_ROOT / "data"


def test_relative_data_dir_resolves_against_repo_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", "some_rel_dir")
    get_settings.cache_clear()
    monkeypatch.chdir(tmp_path)
    assert resolve_data_root(get_settings()) == (REPO_ROOT / "some_rel_dir").resolve()


def test_canonical_paths_identical_from_different_working_directories(tmp_path):
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from retrieval.index_manifest import IndexStore;"
        "print(IndexStore().root)" % (REPO_ROOT / "backend")
    )
    outs = []
    for cwd in (tmp_path, REPO_ROOT, REPO_ROOT / "backend"):
        env = {**os.environ, "DATA_DIR": str(tmp_path / "canon")}
        r = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip().splitlines()[-1])
    assert len(set(outs)) == 1
    assert Path(outs[0]) == (tmp_path / "canon" / "indexes").resolve()


# ── Diagnostics ───────────────────────────────────────────────────────────────

def test_diagnostics_expose_identity_but_no_paths(tmp_path):
    store = _store(tmp_path)
    _mock_build(store)
    d = store.diagnostics(MockEmbeddingProvider(dimension=384))
    assert d["compatible"] is True
    assert d["embedding"]["model"] == "mock-minilm-384" and d["embedding"]["is_mock"] is True
    assert d["generation_id"].startswith("gen-")
    blob = json.dumps(d)
    assert str(tmp_path) not in blob and "\\\\" not in blob.replace("\\\\n", "")
    d2 = store.diagnostics(OtherModelProvider("all-MiniLM-L6-v2", 384))
    assert d2["compatible"] is False and d2["reason"] == "embedding_model_mismatch"


@pytest.mark.asyncio
async def test_index_status_endpoint(mock_index):
    from httpx import ASGITransport, AsyncClient
    from main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/v1/retrieval/index/status")
    assert r.status_code == 200
    body = r.json()
    assert body["compatible"] is True and body["embedding"]["is_mock"] is True


@pytest.mark.asyncio
async def test_api_returns_503_for_integrity_error_not_500(tmp_path):
    from httpx import ASGITransport, AsyncClient
    from main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/v1/retrieval/search", json={"query": "rag", "mode": "dense"})
    assert r.status_code == 503
    assert r.json()["code"] in {"index_manifest_missing", "embedding_model_load_failed"}

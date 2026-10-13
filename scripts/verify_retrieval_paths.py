"""
Phase 8.6 verification of the four retrieval paths against the ACTIVE index.

Independently of the retrievers' own code it checks that:
  1. BM25 returns lexical scores from the active generation's BM25 store.
  2. Dense scores equal cosine similarity recomputed from the STORED vectors with a
     freshly loaded real embedder (so it is neither BM25 nor mock vectors).
  3. Hybrid carries both ranks and reports the embedding model + generation.
  4. The neural cross-encoder actually runs (not fallback) and changes the scores.
  5. A mock provider is rejected against this index (fail-closed).

Usage:  python scripts/verify_retrieval_paths.py [--output reports/evaluation]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from core.config import get_settings  # noqa: E402
from embedding.factory import create_embedding_provider  # noqa: E402
from embedding.identity import describe_embedding_provider  # noqa: E402
from embedding.mock_provider import MockEmbeddingProvider  # noqa: E402
from ranking.cross_encoder_reranker import CrossEncoderReranker  # noqa: E402
from retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from retrieval.dense_retriever import DenseRetriever  # noqa: E402
from retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from retrieval.index_manifest import IndexIntegrityError, IndexStore  # noqa: E402
from retrieval.models import RetrievalQuery  # noqa: E402

QUERIES = [
    ("What is retrieval-augmented generation?", "2005.11401"),
    ("dual encoder dense passage retrieval for question answering", "2004.04906"),
    ("self-attention without recurrence", "1706.03762"),
    ("siamese BERT sentence embeddings", "1908.10084"),
    ("reciprocal rank fusion", "0905.0001"),
]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="reports/evaluation")
    args = ap.parse_args()

    st = get_settings()
    store = IndexStore(st)
    provider = create_embedding_provider(st)
    manifest = store.validate_for_read(provider)  # raises (fail closed) if incompatible
    identity = describe_embedding_provider(provider)

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    check("manifest is complete & real", manifest.status == "complete" and not manifest.embedding.is_mock,
          f"model={manifest.embedding.model_name} dim={manifest.embedding.dimension}")
    integrity = store.verify_integrity()
    check("deep integrity (bm25==qdrant ids, vectors valid, tags)", integrity["ok"],
          f"docs={integrity['documents']} chunks={integrity['chunks']}")

    # Stored vectors, read directly from embedded Qdrant
    stored: dict[str, tuple[list[float], dict]] = {}
    with store.qdrant(manifest.generation_id, manifest.embedding.dimension) as qc:
        pts, _ = qc.scroll(collection_name=store.collection_name, limit=1000, with_payload=True, with_vectors=True)
        for p in pts:
            stored[str(p.id)] = (list(p.vector), p.payload or {})

    bm25 = BM25Retriever(settings=st, index_store=store)
    bm25._server_available = False
    dense = DenseRetriever(settings=st, embedding_provider=provider, index_store=store)
    dense._server_available = False
    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense, settings=st)
    reranker = CrossEncoderReranker(settings=st)
    reranker_loaded = reranker.warmup()

    per_query = []
    max_abs_err = 0.0
    dense_top1_correct = bm25_top1_correct = hybrid_top1_correct = rerank_top1_correct = 0
    rerank_changed_scores = True
    rerank_fallback_seen = False
    for text, gold in QUERIES:
        q = RetrievalQuery(query_text=text, top_k=7)
        b = await bm25.search(q)
        d = await dense.search(q)
        h = await hybrid.search(q)
        qv = provider.embed_query(text)
        # independent cosine from stored vectors
        for c in d.chunks:
            vec, _ = stored[c.chunk_id]
            expected = sum(a * b_ for a, b_ in zip(qv, vec))
            max_abs_err = max(max_abs_err, abs(expected - c.dense_score))
        rr = await reranker.rerank(text, h.chunks, top_k=7)
        rerank_fallback_seen |= any(c.metadata.get("reranker_fallback") for c in rr)
        rerank_changed_scores &= all(abs((c.rerank_score or 0.0) - float(c.metadata.get("original_score", 0.0))) > 1e-6 for c in rr)
        bm25_top1_correct += int(b.chunks[0].source_id == gold)
        dense_top1_correct += int(d.chunks[0].source_id == gold)
        hybrid_top1_correct += int(h.chunks[0].source_id == gold)
        rerank_top1_correct += int(rr[0].source_id == gold)
        per_query.append(
            {
                "query": text,
                "gold": gold,
                "bm25_top": [(c.source_id, c.bm25_score) for c in b.chunks[:3]],
                "dense_top": [(c.source_id, round(c.dense_score, 4)) for c in d.chunks[:3]],
                "hybrid_top": [(c.source_id, c.score) for c in h.chunks[:3]],
                "rerank_top": [(c.source_id, round(c.rerank_score, 3)) for c in rr[:3]],
            }
        )

    n = len(QUERIES)
    check("BM25 results carry lexical scores from active generation",
          all(c.get("fallback") == "local_bm25_store" and c.get("index_generation") == manifest.generation_id
              for c in [(await bm25.search(RetrievalQuery(query_text="rank fusion", top_k=3))).metadata]),
          f"top1 {bm25_top1_correct}/{n}")
    check("dense scores == cosine(stored vector, real query embedding)", max_abs_err < 1e-4,
          f"max|err|={max_abs_err:.2e}; top1 {dense_top1_correct}/{n}")
    d0 = await dense.search(RetrievalQuery(query_text="rank fusion", top_k=3))
    check("dense metadata reports real embedder + generation",
          d0.metadata.get("embedding_model") == identity.model_name
          and d0.metadata.get("index_generation") == manifest.generation_id
          and d0.metadata.get("embedding_is_mock") is False)
    h0 = await hybrid.search(RetrievalQuery(query_text="rank fusion", top_k=5))
    check("hybrid fuses both rankings",
          any(c.bm25_rank is not None and c.dense_rank is not None for c in h0.chunks)
          and h0.metadata.get("embedding_model") == identity.model_name,
          f"top1 {hybrid_top1_correct}/{n}")
    check("neural cross-encoder loaded (not fallback)", reranker_loaded and reranker._model is not None,
          reranker.model_name)
    check("rerank scores are model logits, no fallback flag", (not rerank_fallback_seen) and rerank_changed_scores,
          f"top1 {rerank_top1_correct}/{n}")

    try:
        store.validate_for_read(MockEmbeddingProvider(dimension=384))
        check("mock provider rejected by real index", False, "mock was ACCEPTED")
    except IndexIntegrityError as e:
        check("mock provider rejected by real index", True, e.code)

    ok = all(c["passed"] for c in checks)
    report = {
        "phase": "8.6",
        "timestamp": datetime.now(UTC).isoformat(),
        "overall": "PASS" if ok else "FAIL",
        "embedding": identity.model_dump(),
        "index_generation": manifest.generation_id,
        "documents": integrity["documents"],
        "chunks": integrity["chunks"],
        "reranker_model": reranker.model_name,
        "reranker_mode": "neural" if reranker_loaded else "fallback",
        "dense_score_max_abs_error": max_abs_err,
        "checks": checks,
        "per_query": per_query,
        "note": "5 smoke queries over a 7-chunk corpus; a wiring/identity check, not a quality benchmark.",
    }
    out = _ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase86_retrieval_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nOVERALL: {report['overall']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

"""
Phase 11 latency measurement (no LLM calls, no Gemini quota).

  components : cold model initialization vs warm inference for the cross-encoder and the
               embedding model, using real passages from the ACTIVE index.
  --api URL  : N sequential POST /research/answer requests against a running backend
               (the first request is reported separately as cold). The LLM behind the API
               is whatever the backend is configured with; the report records the
               provider/model from the responses so a mock run cannot be mistaken for real.

Usage:
  python scripts/measure_latency.py --components
  python scripts/measure_latency.py --api http://localhost:8000/api/v1 --n 8
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

QUERIES = [
    "What is retrieval-augmented generation?",
    "How does dense passage retrieval compare with BM25?",
    "Why use reciprocal rank fusion in hybrid search?",
    "How do sentence embeddings enable semantic similarity search?",
    "What are cross-encoders and why are they slower than bi-encoders?",
]


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(round(p * (len(s) - 1))))], 1)


def components() -> dict:
    from core.config import get_settings
    from retrieval.index_manifest import IndexStore
    from retrieval.local_bm25 import LocalBM25Store

    st = get_settings()
    store = IndexStore(st)
    m = store.active_manifest()
    chunks = list(LocalBM25Store(storage_file=store.bm25_path(m.generation_id)).chunks.values())
    texts = [c.content for c in chunks][:25]
    out: dict = {"index_generation": m.generation_id, "passages_used": len(texts)}

    # Embedding model
    from sentence_transformers import SentenceTransformer

    t = time.perf_counter()
    emb = SentenceTransformer(st.embedding_model)
    out["embedding_model_init_ms"] = round((time.perf_counter() - t) * 1000, 1)
    t = time.perf_counter()
    emb.encode([QUERIES[0]], normalize_embeddings=True)
    out["embedding_first_query_ms"] = round((time.perf_counter() - t) * 1000, 1)
    warm = []
    for q in QUERIES * 4:
        t = time.perf_counter()
        emb.encode([q], normalize_embeddings=True)
        warm.append((time.perf_counter() - t) * 1000)
    out["embedding_warm_query_ms"] = {"n": len(warm), "p50": pct(warm, 0.5), "p95": pct(warm, 0.95)}

    # Cross-encoder
    from sentence_transformers import CrossEncoder

    t = time.perf_counter()
    ce = CrossEncoder(st.reranker_model, max_length=st.reranker_max_length, device=st.reranker_device)
    out["reranker_model_init_ms"] = round((time.perf_counter() - t) * 1000, 1)
    pairs = [[QUERIES[0], x] for x in texts]
    t = time.perf_counter()
    ce.predict(pairs, batch_size=st.reranker_batch_size, show_progress_bar=False)
    out["reranker_first_inference_25_pairs_ms"] = round((time.perf_counter() - t) * 1000, 1)
    warm = []
    for i in range(10):
        pr = [[QUERIES[i % len(QUERIES)], x] for x in texts]
        t = time.perf_counter()
        ce.predict(pr, batch_size=st.reranker_batch_size, show_progress_bar=False)
        warm.append((time.perf_counter() - t) * 1000)
    out["reranker_warm_inference_25_pairs_ms"] = {
        "n": len(warm), "p50": pct(warm, 0.5), "p95": pct(warm, 0.95),
        "per_pair_p50": round(pct(warm, 0.5) / max(len(texts), 1), 1),
    }
    out["reranker_model"] = st.reranker_model
    out["device"] = st.reranker_device
    return out


def api(base: str, n: int) -> dict:
    import httpx

    lat, meta = [], []
    with httpx.Client(timeout=200) as c:
        for i in range(n):
            q = QUERIES[i % len(QUERIES)]
            t = time.perf_counter()
            r = c.post(
                f"{base}/research/answer",
                json={"query": q, "enable_decomposition": False, "evidence_k": 4, "candidate_k": 25},
            )
            ms = (time.perf_counter() - t) * 1000
            r.raise_for_status()
            b = r.json()
            lat.append(ms)
            meta.append((b["metadata"]["provider"], b["metadata"]["model"],
                         b["metadata"]["retrieval_latency_ms"], b["metadata"]["generation_latency_ms"]))
    warm = lat[1:]
    return {
        "n_requests": n,
        "provider": meta[0][0], "model": meta[0][1],
        "first_request_ms (cold)": round(lat[0], 1),
        "warm_total_ms": {"n": len(warm), "p50": pct(warm, 0.5), "p95": pct(warm, 0.95), "max": round(max(warm), 1)},
        "warm_retrieval_stage_ms_p50": pct([m[2] for m in meta[1:]], 0.5),
        "warm_generation_stage_ms_p50": pct([m[3] for m in meta[1:]], 0.5),
        "all_ms": [round(x, 1) for x in lat],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", action="store_true")
    ap.add_argument("--api", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--output", default="reports/evaluation/phase11")
    a = ap.parse_args()
    res: dict = {"timestamp": datetime.now(UTC).isoformat(), "host_cpu_only": True}
    if a.components:
        res["components"] = components()
    if a.api:
        res["api"] = api(a.api, a.n)
    out = _ROOT / a.output
    out.mkdir(parents=True, exist_ok=True)
    name = "latency_components.json" if a.components and not a.api else "latency_api.json"
    (out / name).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

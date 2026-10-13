"""
Phase 11: reproducible, modest corpus expansion with distractor papers.

Fetches a FIXED list of public arXiv papers (title + abstract) through the existing
ArxivConnector, one request per paper with >=3.2 s spacing (arXiv guidance), and writes
`data/corpus/corpus.json` = original 7-document corpus + fetched documents.

Selection policy: the papers are well-known general ML/NLP/vision papers chosen to be
OFF-TOPIC for the benchmark's labelled topics (RAG, dense/hybrid retrieval, SBERT, RRF,
cross-encoder reranking, scaling laws). They act as distractors. Because benchmark labels
were written for a 7-document corpus, some added papers (e.g. BERT/T5 for "transformer"
questions) can be topically adjacent yet unlabeled; scores are therefore conservative.

Usage:  python scripts/build_expanded_corpus.py
Never overwrites the original: it is preserved as corpus_v1_7docs.json.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from connectors.arxiv_connector import ArxivConnector  # noqa: E402

DISTRACTOR_IDS = [
    "1810.04805", "1907.11692", "2005.14165", "1910.10683", "1802.05365", "1301.3781",
    "1412.6980", "1512.03385", "2010.11929", "1406.2661", "2006.11239", "2103.00020",
    "1502.03167", "1409.3215", "1409.0473", "2203.02155", "2201.11903", "2302.13971",
    "2307.09288", "2305.18290", "2106.09685", "2312.00752", "1609.02907", "1506.02640",
    "1505.04597", "1312.5602", "1707.06347", "1609.03499", "2004.05150", "2101.03961",
    "2204.02311", "2210.11416", "2302.04761", "2303.08774",
]


async def main() -> int:
    corpus_dir = _ROOT / "data" / "corpus"
    current = corpus_dir / "corpus.json"
    v1 = corpus_dir / "corpus_v1_7docs.json"
    if not v1.exists():
        shutil.copy2(current, v1)
    base = json.loads(v1.read_text(encoding="utf-8"))
    docs = list(base["documents"])
    have = {d["source_id"] for d in docs}

    conn = ArxivConnector()
    fetched, failed = [], []
    for i, aid in enumerate(DISTRACTOR_IDS):
        if aid in have:
            continue
        doc = await conn.fetch(aid)
        if doc is None:
            failed.append(aid)
            print(f"[{i + 1}/{len(DISTRACTOR_IDS)}] {aid}: FAILED")
        else:
            req = doc.to_ingest_request()
            docs.append(
                {
                    "source_type": "arxiv",
                    "source_id": aid,
                    "title": req.title,
                    "url": req.url,
                    "content": req.content,
                    "metadata": req.metadata,
                    "provenance": f"arxiv_api fetched {datetime.now(UTC).date().isoformat()} (distractor)",
                }
            )
            fetched.append(aid)
            print(f"[{i + 1}/{len(DISTRACTOR_IDS)}] {aid}: {req.title[:60]}")
        await asyncio.sleep(3.2)

    payload = {
        "description": base.get("description", "") + " Phase 11: extended with arXiv distractor papers.",
        "documents": docs,
    }
    current.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"total_documents": len(docs), "fetched": len(fetched), "failed": failed}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

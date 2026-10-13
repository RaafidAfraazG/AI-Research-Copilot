"""
Index administration CLI (Phase 8.6).

    python scripts/index_admin.py inventory            # read-only: legacy + canonical stores
    python scripts/index_admin.py backup               # snapshot legacy stores into <data>/backups/
    python scripts/index_admin.py build-corpus         # assemble <data>/corpus/corpus.json
    python scripts/index_admin.py rebuild              # build+verify+activate a new generation
    python scripts/index_admin.py verify               # deep integrity check of the active generation
    python scripts/index_admin.py status               # manifest summary + compatibility
    python scripts/index_admin.py generations          # list generations
    python scripts/index_admin.py prune                # remove incomplete generations

Works from any directory. Never writes outside the canonical data root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from core.config import get_settings  # noqa: E402
from core.paths import backups_root, corpus_file, resolve_data_root  # noqa: E402
from retrieval.index_admin import (  # noqa: E402
    backup_paths,
    inventory_legacy_store,
    list_generation_status,
    load_corpus,
    prune_incomplete,
    rebuild_index,
)
from retrieval.index_manifest import IndexStore  # noqa: E402

# Legacy locations that predate the manifest/generation layout.
LEGACY_STORES = [_ROOT / "data", _ROOT / "backend" / "data"]

# Documents that are labeled as relevant in data/evaluation/retrieval_benchmark.json but
# are not part of data/sample_papers.json. They are recovered verbatim (single-chunk
# documents) from the legacy index so the benchmark keeps its labels unchanged.
RECOVER_FROM_LEGACY = {"2608.09650", "test.integration.001"}


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_inventory(_: argparse.Namespace) -> int:
    st = get_settings()
    out = {
        "canonical_data_root": str(resolve_data_root(st)),
        "legacy_stores": [inventory_legacy_store(p) for p in LEGACY_STORES],
        "generations": list_generation_status(IndexStore(st)),
        "active_generation": IndexStore(st).active_generation_id(),
    }
    _print(out)
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    st = get_settings()
    paths: list[Path] = []
    for root in LEGACY_STORES:
        for sub in ("opensearch_local", "qdrant_local"):
            p = root / sub
            if p.exists():
                # keep the two legacy roots distinguishable inside the backup
                paths.append(p)
    dest_root = backups_root(st)
    dest_root.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    for root in LEGACY_STORES:
        existing = [root / s for s in ("opensearch_local", "qdrant_local") if (root / s).exists()]
        if not existing:
            continue
        label = "legacy_" + ("root_data" if root == _ROOT / "data" else "backend_data")
        res = backup_paths(existing, dest_root, label=label)
        result[label] = {"dir": str(res["backup_dir"].relative_to(dest_root.parent)), "files": res["file_count"]}
    _print(result)
    return 0


def cmd_build_corpus(args: argparse.Namespace) -> int:
    st = get_settings()
    out_path = corpus_file(st)
    sample = _ROOT / "data" / "sample_papers.json"
    papers = json.loads(sample.read_text(encoding="utf-8"))
    documents = []
    for p in papers:
        documents.append(
            {
                "source_type": p.get("source_type", "arxiv"),
                "source_id": p["source_id"],
                "title": p.get("title", ""),
                "url": p.get("url"),
                "content": p["content"],
                "metadata": p.get("metadata", {}),
                "provenance": "data/sample_papers.json",
            }
        )
    legacy_bm25 = Path(args.legacy_bm25) if args.legacy_bm25 else _ROOT / "data" / "opensearch_local" / "bm25_index.json"
    if not legacy_bm25.exists():
        # after the first run the legacy path may be a backup copy
        print(f"legacy BM25 file not found: {legacy_bm25}", file=sys.stderr)
        return 2
    legacy = json.loads(legacy_bm25.read_text(encoding="utf-8")).get("chunks", [])
    have = {d["source_id"] for d in documents}
    for sid in sorted(RECOVER_FROM_LEGACY - have):
        rows = [c for c in legacy if c.get("source_id") == sid]
        texts = {c["content"] for c in rows}
        if len(texts) != 1:
            print(f"cannot recover {sid}: {len(texts)} distinct chunk texts", file=sys.stderr)
            return 2
        c = rows[0]
        documents.append(
            {
                "source_type": c.get("source_type", "arxiv"),
                "source_id": sid,
                "title": c.get("title", ""),
                "url": c.get("url"),
                "content": c["content"],
                "metadata": c.get("metadata", {}),
                "provenance": "recovered_from_legacy_index (single unique chunk; duplicates collapsed)",
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Authoritative source documents for index builds. Excludes test-suite pollution "
            "found in the legacy index (e.g. 'test-doc-1' and unit-test fixture strings)."
        ),
        "documents": documents,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _print({"corpus_documents": len(documents), "ids": [d["source_id"] for d in documents]})
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    from embedding.factory import create_embedding_provider

    st = get_settings()
    docs, sha = load_corpus(Path(args.corpus) if args.corpus else corpus_file(st))
    provider = create_embedding_provider(st)  # mock only if EMBEDDING_PROVIDER=mock
    store = IndexStore(st)
    report = rebuild_index(
        store, provider, docs, corpus_sha256=sha, settings=st,
        allow_mock=args.allow_mock, activate=not args.no_activate, progress=print,
    )
    _print(report)
    return 0 if report["status"] == "complete" else 1


def cmd_verify(_: argparse.Namespace) -> int:
    rep = IndexStore(get_settings()).verify_integrity()
    _print(rep)
    return 0 if rep["ok"] else 1


def cmd_status(_: argparse.Namespace) -> int:
    from embedding.factory import create_embedding_provider

    st = get_settings()
    _print(IndexStore(st).diagnostics(create_embedding_provider(st)))
    return 0


def cmd_generations(_: argparse.Namespace) -> int:
    _print(list_generation_status(IndexStore(get_settings())))
    return 0


def cmd_prune(_: argparse.Namespace) -> int:
    _print({"removed": prune_incomplete(IndexStore(get_settings()))})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory").set_defaults(fn=cmd_inventory)
    sub.add_parser("backup").set_defaults(fn=cmd_backup)
    bc = sub.add_parser("build-corpus")
    bc.add_argument("--legacy-bm25", default=None)
    bc.set_defaults(fn=cmd_build_corpus)
    rb = sub.add_parser("rebuild")
    rb.add_argument("--corpus", default=None)
    rb.add_argument("--allow-mock", action="store_true", help="experiments only; never for the real index")
    rb.add_argument("--no-activate", action="store_true")
    rb.set_defaults(fn=cmd_rebuild)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("generations").set_defaults(fn=cmd_generations)
    sub.add_parser("prune").set_defaults(fn=cmd_prune)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

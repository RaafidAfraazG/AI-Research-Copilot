"""
CLI Script to Ingest External Papers/Documents into the Retrieval Pipeline.

Usage:
    python scripts/ingest_source.py arxiv "cross-encoder reranking" --limit 3
    python scripts/ingest_source.py semantic_scholar "retrieval augmented generation" --limit 3
    python scripts/ingest_source.py web "https://en.wikipedia.org/wiki/Vector_space_model"
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from connectors.ingestion_service import ConnectorIngestionService
from db.database import get_session_factory


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest research documents from external sources")
    parser.add_argument("source", choices=["arxiv", "semantic_scholar", "web"], help="Source connector")
    parser.add_argument("target", help="Search query, paper ID (e.g. 2005.11401), or URL")
    parser.add_argument("--limit", type=int, default=3, help="Max results to ingest")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Ingesting from [{args.source}] -> '{args.target}' (Limit: {args.limit})")
    print("=" * 70)

    service = ConnectorIngestionService()
    session_factory = get_session_factory()

    try:
        if args.source == "web":
            async with session_factory() as session:
                res = await service.fetch_and_ingest("web", args.target, db_session=session)
            if res:
                print(f"\n[OK] Web document indexed: {res.document_id} -> {res.total_chunks} chunks ({res.message})")
            else:
                print(f"\n[FAIL] Could not fetch web document: {args.target}")
        elif ("/" in args.target or "." in args.target) and len(args.target) < 25:
            # Single paper ID fetch
            async with session_factory() as session:
                res = await service.fetch_and_ingest(args.source, args.target, db_session=session)
            if res:
                print(f"\n[OK] Paper indexed: {res.document_id} -> {res.total_chunks} chunks ({res.message})")
            else:
                print(f"\n[FAIL] Could not fetch paper: {args.target}")
        else:
            # Query search and ingest
            async with session_factory() as session:
                results = await service.search_and_ingest(args.source, args.target, limit=args.limit, db_session=session)
            print(f"\n[OK] Successfully ingested {len(results)} documents:")
            for r in results:
                print(f"  - [{r.status}] {r.source_type}:{r.source_id} -> {r.total_chunks} chunks ({r.message})")

    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

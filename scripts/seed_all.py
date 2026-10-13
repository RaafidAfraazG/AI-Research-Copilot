"""
Unified seeding script for AI Research Copilot.
Seeds sample corpus into PostgreSQL, OpenSearch, and Qdrant simultaneously.

Usage:
    python scripts/seed_all.py [--recreate]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from core.config import get_settings
from db.database import get_session_factory
from retrieval.indexing_service import IndexingService
from retrieval.models import IngestDocumentRequest


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed sample research corpus across all datastores")
    parser.add_argument("--recreate", action="store_true", help="Recreate index and collection")
    args = parser.parse_args()

    settings = get_settings()
    data_file = _ROOT / "data" / "sample_papers.json"
    if not data_file.exists():
        print(f"Error: Sample data file not found at {data_file}")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    service = IndexingService(settings=settings)
    print("Initializing OpenSearch index and Qdrant collection...")
    try:
        await service.initialize_stores(recreate=args.recreate)
        print("[OK] OpenSearch & Qdrant initialized")
    except Exception as e:
        print(f"! Notice: Could not initialize remote stores ({e}). Proceeding...")

    session_factory = get_session_factory()
    print(f"\nIngesting {len(papers)} sample research documents...")

    for p in papers:
        req = IngestDocumentRequest(
            source_type=p.get("source_type", "arxiv"),
            source_id=p.get("source_id", ""),
            title=p.get("title", ""),
            content=p.get("content", ""),
            url=p.get("url"),
            metadata=p.get("metadata", {}),
        )
        try:
            async with session_factory() as session:
                async with session.begin():
                    res = await service.index_document(req, db_session=session)
                    print(f"  [{res.status}] '{req.title}' -> {res.total_chunks} chunks ({res.message})")
        except Exception as e:
            # If PostgreSQL is offline, index into search engines only
            res = await service.index_document(req, db_session=None)
            print(f"  [{res.status}] '{req.title}' -> {res.total_chunks} chunks (without DB: {e})")

    try:
        await service.opensearch_client.close()
        await service.qdrant_client.close()
    except Exception:
        pass

    print("\n[OK] Seeding finished successfully!")


if __name__ == "__main__":
    asyncio.run(main())

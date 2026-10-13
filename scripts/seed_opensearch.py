"""
Seed OpenSearch index with sample research papers.
Usage:
    python scripts/seed_opensearch.py [--recreate]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

# Add backend directory to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from core.config import get_settings
from retrieval.indexing_service import IndexingService
from retrieval.models import IngestDocumentRequest
from retrieval.opensearch_schema import ensure_opensearch_index


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed OpenSearch with sample documents")
    parser.add_argument("--recreate", action="store_true", help="Recreate index from scratch")
    args = parser.parse_args()

    settings = get_settings()
    data_file = _ROOT / "data" / "sample_papers.json"
    if not data_file.exists():
        print(f"Sample data file not found: {data_file}")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    service = IndexingService(settings=settings)
    print(f"Ensuring OpenSearch index '{settings.opensearch_index_name}' exists...")
    await ensure_opensearch_index(
        client=service.opensearch_client,
        index_name=settings.opensearch_index_name,
        recreate=args.recreate,
    )

    print(f"Indexing {len(papers)} documents into OpenSearch...")
    for p in papers:
        req = IngestDocumentRequest(
            source_type=p.get("source_type", "arxiv"),
            source_id=p.get("source_id", ""),
            title=p.get("title", ""),
            content=p.get("content", ""),
            url=p.get("url"),
            metadata=p.get("metadata", {}),
        )
        res = await service.index_document(req)
        print(f"  [{res.status}] '{req.title}' -> {res.total_chunks} chunks")

    await service.opensearch_client.close()
    print("OpenSearch seeding complete!")


if __name__ == "__main__":
    asyncio.run(main())

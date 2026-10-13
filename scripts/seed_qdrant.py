"""
Seed Qdrant vector database with sample research papers.
Usage:
    python scripts/seed_qdrant.py [--recreate]
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
from retrieval.qdrant_schema import ensure_qdrant_collection


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Qdrant with sample document embeddings")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection from scratch")
    args = parser.parse_args()

    settings = get_settings()
    data_file = _ROOT / "data" / "sample_papers.json"
    if not data_file.exists():
        print(f"Sample data file not found: {data_file}")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    service = IndexingService(settings=settings)
    print(f"Ensuring Qdrant collection '{settings.qdrant_collection_name}' exists...")
    await ensure_qdrant_collection(
        client=service.qdrant_client,
        collection_name=settings.qdrant_collection_name,
        vector_dimension=service.embedding_provider.dimension,
        recreate=args.recreate,
    )

    print(f"Indexing {len(papers)} documents into Qdrant...")
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

    await service.qdrant_client.close()
    print("Qdrant seeding complete!")


if __name__ == "__main__":
    asyncio.run(main())

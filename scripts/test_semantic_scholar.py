"""
CLI Test Script for Semantic Scholar Connector.
Usage:
    python scripts/test_semantic_scholar.py "dense passage retrieval" [--limit 5]
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from connectors.models import ConnectorSearchRequest
from connectors.semantic_scholar_connector import SemanticScholarConnector


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test Semantic Scholar source connector")
    parser.add_argument("query", nargs="?", default="dense passage retrieval", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Maximum papers to return")
    parser.add_argument("--year", type=int, default=None, help="Filter minimum publication year")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Searching Semantic Scholar for: '{args.query}' (Limit: {args.limit})")
    print("=" * 70)

    connector = SemanticScholarConnector()
    req = ConnectorSearchRequest(query=args.query, limit=args.limit, year=args.year)
    response = await connector.search(req)

    print(f"\n[OK] Found {len(response.documents)} papers in {response.latency_ms}ms (Total indexed: {response.total_results}):\n")

    for i, doc in enumerate(response.documents, start=1):
        print(f"[{i}] {doc.title}")
        print(f"    Paper ID:   {doc.source_id}")
        print(f"    Authors:    {', '.join(doc.authors[:4])}{' et al.' if len(doc.authors) > 4 else ''}")
        print(f"    Year/Venue: {doc.publication_date or 'N/A'} | {doc.venue or 'N/A'}")
        print(f"    Citations:  {doc.citation_info.citation_count if doc.citation_info.citation_count is not None else 'N/A'}")
        print(f"    URL:        {doc.url}")
        if doc.citation_info.tldr:
            print(f"    TLDR:       {doc.citation_info.tldr}")
        elif doc.abstract:
            snippet = doc.abstract[:180].replace("\n", " ") + "..."
            print(f"    Abstract:   {snippet}")
        print("-" * 70)


if __name__ == "__main__":
    asyncio.run(main())

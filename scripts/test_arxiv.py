"""
CLI Test Script for arXiv Connector.
Usage:
    python scripts/test_arxiv.py "retrieval augmented generation" [--limit 5]
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from connectors.arxiv_connector import ArxivConnector
from connectors.models import ConnectorSearchRequest


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test arXiv source connector")
    parser.add_argument("query", nargs="?", default="retrieval augmented generation", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Maximum papers to return")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Searching arXiv for: '{args.query}' (Limit: {args.limit})")
    print("=" * 70)

    connector = ArxivConnector()
    req = ConnectorSearchRequest(query=args.query, limit=args.limit)
    response = await connector.search(req)

    print(f"\n[OK] Found {len(response.documents)} papers in {response.latency_ms}ms:\n")

    for i, doc in enumerate(response.documents, start=1):
        print(f"[{i}] {doc.title}")
        print(f"    arXiv ID:  {doc.source_id}")
        print(f"    Authors:   {', '.join(doc.authors[:4])}{' et al.' if len(doc.authors) > 4 else ''}")
        print(f"    Date:      {doc.publication_date or 'N/A'}")
        print(f"    URL:       {doc.url}")
        print(f"    PDF:       {doc.pdf_url}")
        print(f"    Categories: {', '.join(doc.categories)}")
        if doc.abstract:
            snippet = doc.abstract[:180].replace("\n", " ") + "..."
            print(f"    Abstract:  {snippet}")
        print("-" * 70)


if __name__ == "__main__":
    asyncio.run(main())

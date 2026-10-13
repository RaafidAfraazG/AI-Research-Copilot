"""
CLI Test Script for Web Fetcher.
Usage:
    python scripts/test_web_fetcher.py "https://en.wikipedia.org/wiki/Information_retrieval"
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from connectors.web_fetcher import SSRFValidationError, WebFetcher


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test secure WebFetcher on a target URL")
    parser.add_argument("url", nargs="?", default="https://en.wikipedia.org/wiki/Information_retrieval", help="URL to fetch")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Fetching Web Document: '{args.url}'")
    print("=" * 70)

    fetcher = WebFetcher()
    try:
        doc = await fetcher.fetch(args.url)
        if not doc:
            print("Failed to extract content from URL")
            return

        print("\n[OK] Document fetched and extracted successfully:\n")
        print(f"Title:         {doc.title}")
        print(f"Authors:       {', '.join(doc.authors) if doc.authors else 'N/A'}")
        print(f"Canonical URL: {doc.url}")
        print(f"Description:   {doc.abstract or 'N/A'}")
        print(f"Content Size:  {len(doc.full_text or '')} chars")
        if doc.full_text:
            print("\nPreview of extracted content:")
            print("-" * 50)
            preview = doc.full_text[:400].replace("\n\n", "\n")
            print(f"{preview}...")
            print("-" * 50)

    except SSRFValidationError as e:
        print(f"[SECURITY BLOCKED] SSRF Validation failed: {e}")
    except Exception as e:
        print(f"[ERROR] Fetch failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

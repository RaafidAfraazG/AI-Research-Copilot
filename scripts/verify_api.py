"""
FastAPI Live Endpoint Verification Script.
Tests all health and retrieval endpoints against live local infrastructure.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import httpx

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from main import app


async def test_live_api() -> None:
    print("=" * 70)
    print("Testing Live FastAPI Endpoints (Phase 2.5)")
    print("=" * 70)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost:8000",
        timeout=10.0,
    ) as client:
        # 1. Health Check
        print("\n[1] Testing GET /api/v1/health ...")
        res = await client.get("/api/v1/health")
        print(f"    Status: {res.status_code}")
        print(f"    Response: {res.json()}")
        assert res.status_code == 200

        # 2. Ready Check
        print("\n[2] Testing GET /api/v1/ready ...")
        res = await client.get("/api/v1/ready")
        print(f"    Status: {res.status_code}")
        print(f"    Response: {res.json()}")

        # 3. Chunk Preview
        print("\n[3] Testing POST /api/v1/retrieval/chunk ...")
        res = await client.post(
            "/api/v1/retrieval/chunk",
            json={
                "text": "Dense Passage Retrieval uses dual encoders. BM25 uses TF-IDF matching. Hybrid retrieval combines both.",
                "chunk_size_tokens": 100,
                "chunk_overlap_tokens": 20,
                "title": "API Test Doc",
            },
        )
        print(f"    Status: {res.status_code}")
        print(f"    Total chunks generated: {res.json().get('total_chunks')}")
        assert res.status_code == 200

        # 4. Search Endpoint (Hybrid)
        print("\n[4] Testing POST /api/v1/retrieval/search (Hybrid Mode) ...")
        res = await client.post(
            "/api/v1/retrieval/search",
            json={
                "query": "retrieval augmented generation",
                "top_k": 3,
                "mode": "hybrid",
                "alpha": 0.5,
            },
        )
        print(f"    Status: {res.status_code}")
        data = res.json()
        print(f"    Retriever: {data.get('retriever')}, Latency: {data.get('latency_ms')}ms")
        print(f"    Found {len(data.get('chunks', []))} chunks:")
        for c in data.get("chunks", []):
            print(f"      - [{c.get('score')}] '{c.get('title')}'")
        assert res.status_code == 200
        assert len(data.get("chunks", [])) > 0

        # 5. Index Endpoint
        print("\n[5] Testing POST /api/v1/retrieval/index ...")
        res = await client.post(
            "/api/v1/retrieval/index",
            json={
                "source_type": "local",
                "source_id": "test-doc-1",
                "title": "FastAPI Live Indexing Verification",
                "content": "This document verifies that live indexing via the FastAPI endpoint operates successfully on local infrastructure.",
            },
        )
        print(f"    Status: {res.status_code}")
        print(f"    Response: {res.json()}")
        assert res.status_code == 201

    print("\n" + "=" * 70)
    print("All API endpoints verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_live_api())

# Local Development Guide (Non-Docker Windows Environment)

This guide documents how to run and develop **AI Research Copilot** directly on a Windows development machine without requiring Docker.

---

> **Status note (Phase 12):** this guide predates the index-generation layout. For current setup, run commands and troubleshooting use the root [README](../README.md). In the running setup OpenSearch, a Qdrant server and Redis are **not used**; BM25 and vectors live in `data/indexes/<generation>/` (use `scripts/index_admin.py`). The `data/opensearch_local` / `data/qdrant_local` paths below are legacy. Passwords shown are the template defaults - set your own in `.env`.


## 1. Environment Audit & Prerequisites

| Component | Status on Local Machine | Version Detected | Role in AI Research Copilot |
|---|---|---|---|
| **Python** | Installed | 3.12.10 | Backend API, Retrieval Core, AI/ML pipelines |
| **Node.js & npm** | Installed | Node v20.19.4 / npm 11.5.2 | Frontend React application (Phase 7) |
| **Java** | Installed | 24.0.1 (Oracle HotSpot 64-Bit) | Runtime for optional OpenSearch standalone |
| **PostgreSQL** | Running Service | 17.5 (port **5433**) | Metadata DB: query history, document registry |
| **OpenSearch** | Local Fallback Active | Port 9200 (optional standalone) | Lexical / BM25 keyword search |
| **Qdrant** | Local Embedded Active | Port 6333 (optional standalone) | Dense vector / semantic similarity search |
| **Redis** | Optional (Phase 6+) | Not required for retrieval | Background jobs / query caching |

---

## 2. PostgreSQL Configuration

The machine has PostgreSQL 17 running as a Windows service on port **5433**.

* **Database**: `ai_research_copilot`
* **User**: `copilot_user`
* **Password**: `change_me_in_production`
* **Port**: `5433`

### Run Migrations

To apply database migrations to local PostgreSQL:

```powershell
cd backend
alembic upgrade head
```

---

## 3. Retrieval Architecture: Dual-Mode Operation

The Phase 2 retrieval pipeline is designed for **zero-friction local development** while maintaining complete production compatibility:

```
                  ┌─────────────────────────────────────────┐
                  │          FastAPI Backend                │
                  └──────────────────┬──────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
     ┌───────────────────────┐               ┌───────────────────────┐
     │     BM25Retriever     │               │    DenseRetriever     │
     └───────────┬───────────┘               └───────────┬───────────┘
                 │                                       │
         ┌───────┴───────┐                       ┌───────┴───────┐
         ▼               ▼                       ▼               ▼
   [OpenSearch]   [LocalBM25Store]         [Qdrant Server] [Local Qdrant]
    (Port 9200)   (data/opensearch_local)   (Port 6333)    (data/qdrant_local)
```

1. **OpenSearch (BM25)**:
   * Connects to OpenSearch server on `localhost:9200` if running.
   * If OpenSearch server is offline, it automatically routes queries to `LocalBM25Store` (`data/opensearch_local/bm25_index.json`) with identical Okapi BM25 scoring and field boosting.
2. **Qdrant (Dense)**:
   * Connects to Qdrant cluster on `localhost:6333` if running.
   * If Qdrant cluster is offline, it automatically operates on embedded disk storage in `data/qdrant_local/` using native `qdrant-client` local storage.

---

## 4. Seeding the Sample Research Corpus

To seed the sample AI/ML research papers (`data/sample_papers.json`) across PostgreSQL, OpenSearch (or Local BM25), and Qdrant:

```powershell
python scripts/seed_all.py
```

---

## 5. Running the Backend Server

Start the FastAPI application with Uvicorn:

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

### Endpoints Available:

* **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Liveness Health Check**: `GET http://localhost:8000/api/v1/health`
* **Readiness Check**: `GET http://localhost:8000/api/v1/ready`
* **Multi-Mode Search**: `POST http://localhost:8000/api/v1/retrieval/search`
* **Document Ingestion**: `POST http://localhost:8000/api/v1/retrieval/index`
* **Passage Chunk Preview**: `POST http://localhost:8000/api/v1/retrieval/chunk`

---

## 6. Verifying Retrieval End-to-End

Run the verification scripts:

```powershell
# 1. Test BM25, Dense, and Hybrid (RRF) search
python scripts/verify_retrieval.py

# 2. Test live FastAPI REST endpoints
python scripts/verify_api.py

# 3. Run full automated test suite (Unit + Integration)
python -m pytest backend/tests
```

---

## 7. Optional: Running Standalone Daemons on Windows

If you wish to run dedicated standalone background daemons for OpenSearch or Qdrant without Docker:

### Qdrant Standalone Binary:
1. Download `qdrant-x86_64-pc-windows-msvc.zip` from [Qdrant Releases](https://github.com/qdrant/qdrant/releases).
2. Extract `qdrant.exe` to a local folder (e.g. `C:\tools\qdrant\`).
3. Run `qdrant.exe`. It will start on `http://localhost:6333`.

### OpenSearch Windows Archive:
1. Download `opensearch-2.18.0-windows-x64.zip` from [OpenSearch Downloads](https://opensearch.org/downloads.html).
2. Extract to `C:\tools\opensearch\`.
3. Set `OPENSEARCH_INITIAL_ADMIN_PASSWORD` and run `bin\opensearch.bat`. It will start on `http://localhost:9200`.

---

## 8. Known Windows-Specific Notes

* **Console Unicode**: Windows PowerShell console default code page (cp1252) may reject unicode symbols like `✓`. All scripts use ASCII markers (`[OK]`).
* **PostgreSQL Port**: The local PostgreSQL 17 server is listening on port `5433` (configured in `.env`).
* **Docker**: not used. The project runs without containers; the original `docker-compose.yml` scaffold was removed during repository cleanup because nothing in the setup required it.

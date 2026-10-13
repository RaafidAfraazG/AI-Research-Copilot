# Backend — AI Research Copilot

FastAPI service that plans a research question, retrieves evidence (BM25 + dense, fused with RRF), reranks it with a cross-encoder, synthesizes a cited answer with Gemini, and verifies the citations structurally.

Setup, configuration and the full project overview are in the [root README](../README.md). This file only covers backend-specific commands.

## Requirements

| Tool | Version |
|---|---|
| Python | 3.11+ (developed and tested on 3.12) |
| PostgreSQL | Reachable with the credentials in the root `.env` (metadata + readiness check) |

No Docker, OpenSearch, Qdrant server or Redis is needed: BM25 and the vector index are local files under `data/indexes/`.

## Run

From the **repository root** (the backend reads the root `.env`; there is no `backend/.env`):

```powershell
pip install -r backend/requirements-dev.txt
cd backend
alembic upgrade head                       # create/upgrade PostgreSQL tables
python -m uvicorn main:app --port 8000     # http://localhost:8000, Swagger UI at /docs
```

The vector index must exist before answering questions: `python scripts/index_admin.py rebuild` (from the repository root).

## API

All routes are under `/api/v1`. Interactive reference: `/docs` (Swagger UI) and `/openapi.json`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness; no I/O |
| `GET` | `/ready` | Readiness: PostgreSQL, active index, LLM configuration (no LLM request is made). OpenSearch / Qdrant server / Redis are reported as `skipped` |
| `POST` | `/research/plan` | Query analysis and decomposition |
| `POST` | `/research/retrieve` | Retrieval + rerank + evidence selection (no LLM synthesis) |
| `POST` | `/research/answer` | Full pipeline: cited answer with structural citation verification |
| `POST` | `/retrieval/search` | Single-mode search (BM25, dense or hybrid) |
| `POST` | `/retrieval/index` | Index documents into the active generation |
| `POST` | `/retrieval/chunk` | Passage chunking preview |
| `GET` | `/retrieval/index/status` | Active generation, embedding model, document/chunk counts |
| `GET` | `/sources/arxiv/search`, `/sources/semantic-scholar/search`, `/sources/web/fetch` | Source connectors (the web fetcher is SSRF-hardened) |
| `POST` | `/sources/ingest` | Fetch a source and add it to the active index generation |

Errors use a stable `{error, code, request_id}` body; details are only in redacted logs.

## Tests and linting

```powershell
cd backend
python -m pytest -p no:cacheprovider -q            # ~3 min; needs cached Hugging Face weights, no API key
python -m pytest tests/test_health.py -v           # a single file
$env:RUN_REAL_MODEL_TESTS = "1"; python -m pytest -q tests/test_real_embeddings.py   # opt-in real models

ruff check .
mypy .
```

## Database migrations

[Alembic](https://alembic.sqlalchemy.org/) with async PostgreSQL. Migrations live in `alembic/versions/`.

```powershell
cd backend
alembic upgrade head          # apply all migrations
alembic current               # current revision
alembic revision --autogenerate -m "describe your change"   # after changing ORM models
alembic downgrade -1
```

## Layout

```
backend/
├── main.py              FastAPI app factory + lifespan
├── core/                settings (pydantic-settings), logging, redaction, error handling, data paths
├── api/v1/              health, research, retrieval, sources routers; api/middleware/ timing
├── query/               intent, complexity, decomposition, planning, evidence merging
├── retrieval/           BM25, dense, hybrid (RRF), chunker, index manifest + index admin
├── ranking/             cross-encoder reranker, evidence selection
├── synthesis/           prompt builder, answer service, citation mapping + verification
├── llm/                 Gemini (default), mock, and guarded optional providers
├── embedding/           sentence-transformers (default), mock (tests), identity checks
├── connectors/          arXiv, Semantic Scholar, SSRF-hardened web fetcher, deduplication
├── services/            research retrieval service (orchestration)
├── db/                  SQLAlchemy models and repositories
├── evaluation/          metrics, retrieval/answer/decomposition/system evaluators
├── alembic/             migrations
└── tests/               unit and API tests (+ integration/)
```

## Configuration

All settings come from the root `.env` (template: [`../.env.example`](../.env.example)). `GEMINI_API_KEY` is read by the backend only and must never be placed in frontend configuration.

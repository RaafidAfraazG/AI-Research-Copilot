# ADR 001 — Core Technology Stack Selection

**Status:** Accepted
**Date:** 2026-09-29
**Author:** AI Research Copilot Architecture Team

---

## Context

We are building a production-quality AI research assistant. The technology choices made now will define the development experience, performance ceiling, and operational complexity for the life of the project.

Key constraints:
- Must run entirely locally for development (no mandatory cloud accounts)
- Must support hybrid retrieval (lexical + vector)
- Must be modular and provider-agnostic for LLMs and embeddings
- Must be evaluable (IR metrics built in from the start)
- Python backend preferred for AI/ML ecosystem compatibility
- TypeScript frontend preferred for type safety

---

## Decisions

### Backend Framework: FastAPI

**Alternatives considered:** Flask, Django, Django REST Framework

FastAPI chosen because:
- Native async support matches our async I/O pattern (OpenSearch, Qdrant, LLM APIs all benefit)
- Pydantic v2 integration is first-class (our data contracts are Pydantic models throughout)
- Auto-generated OpenAPI docs reduce API documentation burden
- High performance for an API gateway role

### Lexical Search: OpenSearch (not Elasticsearch)

**Alternatives considered:** Elasticsearch 8.x, Typesense, Meilisearch

OpenSearch chosen because:
- Fully open source (Apache 2.0), no licensing concerns for self-hosting
- API is nearly identical to Elasticsearch 7.10 for our use cases
- Well-maintained, AWS-backed, stable
- Typesense/Meilisearch: insufficient for production-scale BM25 research retrieval

### Vector Database: Qdrant (not Pinecone, Weaviate, pgvector)

**Alternatives considered:** Pinecone, Weaviate, pgvector, ChromaDB, Milvus

Qdrant chosen because:
- Runs locally via Docker (no mandatory cloud account)
- Excellent Python client (`qdrant-client`)
- Supports payload filtering alongside vector search (useful for source_type, date filters)
- Strong performance benchmarks vs alternatives
- ChromaDB/pgvector: appropriate for demos, insufficient at production scale

### Hybrid Fusion: Reciprocal Rank Fusion (not score interpolation)

BM25 scores and cosine similarity scores are incomparable in magnitude. RRF is robust to this, parameter-free (beyond k=60), and empirically competitive. Score interpolation requires careful alpha tuning per query type — deferred as a future optimization.

### Embedding Model: sentence-transformers (not OpenAI embeddings by default)

Default: `all-MiniLM-L6-v2`

Chosen because:
- Runs entirely locally on CPU — no API key, no API cost
- Well-benchmarked on semantic similarity tasks
- Fast enough for moderate-scale indexing
- The `EmbeddingProvider` abstraction allows easy swap to `text-embedding-3-small` etc.

### LLM Abstraction: Direct provider clients + instructor (not LangChain)

**Alternatives considered:** LangChain, LlamaIndex, Haystack

Direct clients chosen because:
- LangChain adds opacity to pipeline stages — debugging retrieval failures is difficult
- Our pipeline stages need explicit data contracts (Pydantic models between stages)
- We need per-stage latency measurement — opaque chains make this hard
- `instructor` library handles structured output cleanly without framework lock-in

### Reranker: Cross-encoder (not bi-encoder, not API-only)

`cross-encoder/ms-marco-MiniLM-L-6-v2` chosen because:
- Runs locally (no API cost)
- Cross-encoders provide higher ranking quality than bi-encoders by jointly encoding query+document
- Applied only to top-M candidates (not full corpus) — acceptable latency

### Application Database: PostgreSQL (not SQLite, not MongoDB)

PostgreSQL chosen because:
- ACID compliant — correct for metadata that must not be lost
- Async SQLAlchemy 2.x support is production-grade
- Easily run via Docker Compose locally
- Full SQL for complex analytical queries on evaluation data

---

## Consequences

### Positive
- Full local development without any cloud accounts (Ollama + Docker)
- Strong type safety throughout (Pydantic, TypeScript)
- Every AI component is swappable via config
- Built-in evaluability from the start

### Negative / Tradeoffs
- OpenSearch has higher RAM requirements than simpler alternatives (~512MB minimum)
- sentence-transformers model download on first run (~90MB)
- Cross-encoder reranking adds latency (~200-500ms for 20 candidates on CPU)
- No LangChain means more explicit pipeline code to write (but this is intentional)

### Future Options Left Open
- Swap to `e5-large-v2` or OpenAI embeddings for higher retrieval quality
- Add Cohere Rerank API for production reranking quality
- Switch Celery to modal.com or Ray for distributed processing
- Add streaming responses from LLM in Phase 7+

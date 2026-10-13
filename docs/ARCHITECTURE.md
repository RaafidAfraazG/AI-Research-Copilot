# Architecture Document — AI Research Copilot

> **Status note (Phase 12):** sections 1-14 below are the original design document and describe a target stack (OpenSearch, Qdrant server, Redis, Celery). The **implemented, running architecture** is the diagram and component table in the root [README](../README.md#architecture), plus sections 4.4 (frontend), 4.5 (runtime hardening) and 5.5 (data root and index generations). Where they differ, README/5.5 are authoritative.


**Version:** 0.1 (Architecture Phase)
**Date:** 2026-09-29
**Status:** Draft

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architectural Principles](#2-architectural-principles)
3. [High-Level Data Flow](#3-high-level-data-flow)
4. [Module Breakdown](#4-module-breakdown)
5. [Data Storage Architecture](#5-data-storage-architecture)
6. [Retrieval Pipeline Design](#6-retrieval-pipeline-design)
7. [AI / ML Components](#7-ai--ml-components)
8. [External Integrations](#8-external-integrations)
9. [LLM Provider Abstraction](#9-llm-provider-abstraction)
10. [Evaluation Strategy](#10-evaluation-strategy)
11. [Technology Decisions & Rationale](#11-technology-decisions--rationale)
12. [Repository Structure](#12-repository-structure)
13. [Phased Implementation Roadmap](#13-phased-implementation-roadmap)
14. [Security & Configuration](#14-security--configuration)
15. [Manual Action Required](#15-manual-action-required)

---

## 1. System Overview

AI Research Copilot is a **retrieval-augmented generation (RAG) system specialized for deep research tasks**. It is distinguished from generic chatbots and simple document Q&A tools by:

- **Query intelligence**: understanding research intent and decomposing it into sub-questions
- **Multi-source retrieval**: pulling from papers, web, code repositories, and custom indexes
- **Hybrid retrieval**: combining lexical (BM25) and dense (vector) search
- **Neural reranking**: using a cross-encoder to refine candidate document ordering
- **Grounded synthesis**: LLM answer generation tightly constrained to retrieved evidence
- **Traceable citations**: every claim linked to a real retrieved document passage
- **Process transparency**: the user can see what was searched and retrieved
- **Evaluability**: retrieval quality metrics are built into the system from the start

The system is **not** a simple LangChain wrapper. It is an explicit, modular pipeline where each stage is independently testable, replaceable, and measurable.

---

## 2. Architectural Principles

| Principle | Rationale |
|---|---|
| **Explicit over magic** | Prefer direct Python modules over opaque framework chains. Each stage of the pipeline should be readable code. |
| **Separation of concerns** | API layer, business logic, retrieval, ranking, LLM orchestration, and data access are distinct layers. |
| **Provider abstraction** | LLM, embeddings, and reranker are behind protocol interfaces. Swapping providers requires config changes, not code changes. |
| **No hardcoded secrets** | All credentials via environment variables. `.env.example` documents what is needed. |
| **Graceful degradation** | If an external source (arXiv, Semantic Scholar) is unavailable, the pipeline continues with remaining sources. |
| **Evaluability by design** | Every retrieval stage records what it returned and its confidence. Evaluation metrics can be computed without retrofitting. |
| **Citations must be real** | The LLM is constrained to only cite documents that were actually retrieved. No hallucinated references. |
| **Local-first development** | All backing services run via Docker Compose. No cloud account is needed for development. |
| **Minimal dependencies** | Each component uses only what it needs. Avoid pulling in heavyweight frameworks for simple tasks. |

---

## 3. High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  USER                                                            │
│  "Compare YOLO vs transformer detectors for edge deployment"     │
└─────────────────────────┬────────────────────────────────────────┘
                          │ HTTP POST /api/v1/research/query
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  FASTAPI — API LAYER                                             │
│  • Request validation (Pydantic)                                 │
│  • Session / query ID assignment                                 │
│  • Async dispatch to Research Orchestrator                       │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  QUERY INTELLIGENCE SERVICE                                      │
│  • Intent classification (research / factual / comparison)       │
│  • Key concept extraction                                        │
│  • Query decomposition → list of focused sub-queries             │
│  • Uses LLM with structured output (Pydantic models)             │
└─────────────────────────┬────────────────────────────────────────┘
                          │ Sub-queries + metadata
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  RETRIEVAL PIPELINE                                              │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │  BM25 Retriever │  │  Dense Retriever  │  │ Source         │  │
│  │  (OpenSearch)   │  │  (Qdrant)         │  │ Connectors     │  │
│  │                 │  │                   │  │ • arXiv        │  │
│  │  Keyword match  │  │  Embedding-based  │  │ • Sem. Scholar │  │
│  │  on indexed docs│  │  cosine similarity│  │ • Web fetcher  │  │
│  └────────┬────────┘  └────────┬──────────┘  └───────┬────────┘  │
│           └──────────┬─────────┘                     │           │
│                      ▼                               │           │
│           ┌──────────────────────┐                   │           │
│           │   Hybrid Fusion      │◄──────────────────┘           │
│           │   (Reciprocal Rank   │                               │
│           │    Fusion — RRF)     │                               │
│           └──────────┬───────────┘                               │
└──────────────────────┼────────────────────────────────────────────┘
                       │ Candidate documents (top-N, e.g. 50)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  RANKING LAYER                                                   │
│  • Cross-encoder reranker scores (query, document) pairs         │
│  • Returns top-K ranked documents (e.g. K=10)                   │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  EVIDENCE EXTRACTION                                             │
│  • Sentence-level passage chunking                               │
│  • Most relevant passage selected per document                   │
│  • Evidence objects include: text, source URL, title, chunk ID   │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  LLM SYNTHESIS                                                   │
│  • Structured prompt: context = evidence passages                │
│  • Instruction: answer ONLY from provided context                │
│  • Output: answer text + inline citation markers                 │
│  • Citation engine maps markers → real source metadata           │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  RESPONSE ASSEMBLY                                               │
│  • Final answer text                                             │
│  • Ordered citations list (title, URL, snippet, relevance score) │
│  • Retrieved sources panel (all candidates)                      │
│  • Pipeline trace (sub-queries used, sources searched)           │
│  • Latency breakdown per stage                                   │
└─────────────────────────┬────────────────────────────────────────┘
                          │ JSON response
                          ▼
                       FRONTEND
```

---

## 4. Module Breakdown

### 4.1 Backend (`backend/`)

```
backend/
├── main.py                    # FastAPI app entry point
├── core/
│   ├── config.py              # Settings via pydantic-settings, reads from .env
│   ├── logging.py             # Structured logging setup (structlog)
│   └── exceptions.py          # Custom exception hierarchy
├── api/
│   ├── v1/
│   │   ├── router.py          # Top-level v1 router
│   │   ├── research.py        # /research/* endpoints
│   │   ├── sources.py         # /sources/* endpoints (manage indexed sources)
│   │   └── health.py          # /health, /ready endpoints
│   └── middleware/
│       ├── timing.py          # Request timing middleware
│       └── cors.py            # CORS config
├── services/
│   ├── research_orchestrator.py  # Coordinates the full pipeline
│   └── query_intelligence.py     # Intent + decomposition service
├── retrieval/
│   ├── base.py                # Abstract Retriever protocol
│   ├── bm25_retriever.py      # OpenSearch BM25 retriever
│   ├── dense_retriever.py     # Qdrant dense retriever
│   ├── hybrid_retriever.py    # RRF fusion of BM25 + dense
│   └── passage_chunker.py     # Document → passage chunking
├── ranking/
│   ├── base.py                # Abstract Reranker protocol
│   └── cross_encoder_reranker.py  # sentence-transformers cross-encoder
├── synthesis/
│   ├── prompt_builder.py      # Constructs LLM prompts from evidence
│   ├── citation_engine.py     # Maps inline markers to source metadata
│   └── answer_generator.py    # Calls LLM, parses structured output
├── connectors/
│   ├── base.py                # Abstract DocumentSource protocol
│   ├── arxiv_connector.py     # Fetches papers from arXiv API
│   ├── semantic_scholar_connector.py
│   ├── web_fetcher.py         # Generic URL fetching + content extraction
│   └── github_connector.py    # GitHub README/code search (future)
├── embedding/
│   ├── base.py                # Abstract EmbeddingProvider protocol
│   └── sentence_transformer_provider.py
├── llm/
│   ├── base.py                # Abstract LLMProvider protocol
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── ollama_provider.py     # Local Ollama for zero-cost dev
├── db/
│   ├── database.py            # Async SQLAlchemy engine setup
│   ├── models/                # ORM models
│   │   ├── query.py           # ResearchQuery model
│   │   ├── document.py        # IndexedDocument model
│   │   └── session.py         # ResearchSession model
│   └── repositories/          # Data access layer (no raw SQL in services)
│       ├── query_repository.py
│       └── document_repository.py
└── evaluation/
    ├── metrics.py             # Recall@K, Precision@K, MRR, NDCG
    └── harness.py             # Evaluation test harness
```

### 4.2 Frontend (`frontend/`)

```
frontend/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── api/                   # API client (TanStack Query hooks)
│   ├── components/
│   │   ├── SearchBar/
│   │   ├── ResultsPanel/
│   │   ├── CitationList/
│   │   ├── ProcessTrace/      # Shows sub-queries, sources searched
│   │   └── ComparisonView/    # Side-by-side comparison mode
│   ├── pages/
│   │   ├── ResearchPage.tsx
│   │   └── SourcesPage.tsx
│   ├── store/                 # Zustand state
│   └── types/                 # Shared TypeScript types
├── index.html
├── vite.config.ts
├── tailwind.config.ts
└── tsconfig.json
```

### 4.3 Evaluation (`evaluation/`)

```
evaluation/
├── datasets/                  # Test query sets with ground-truth relevance
├── scripts/
│   ├── run_eval.py            # Run offline evaluation against a dataset
│   └── generate_queries.py    # LLM-assisted query generation for test sets
└── results/                   # Stored evaluation run outputs
```

---

### 4.4 Frontend implementation (Phase 9)

`frontend/` is a Vite + React + TypeScript + Tailwind single-page app. It contains no
retrieval, ranking or generation logic and holds no secrets; every AI operation is a call to
the FastAPI backend.

```
frontend/src/
  api/client.ts        typed fetch wrapper: base URL (VITE_API_BASE_URL), timeout/abort,
                       error classification -> user-safe messages, response shape guard
  types/api.ts         TypeScript mirror of backend/synthesis/models.py schemas
  lib/answerModel.ts   joins claims <-> citations <-> evidence by backend ids (no invention)
  lib/safeUrl.ts       http(s)-only external links
  components/          Sidebar, Composer, PlanPanel, AnswerView, EvidencePanel, States, Markdown
  App.tsx              state machine: idle -> loading -> success | error
```

Integration contract: one `POST /research/answer` per question. The response supplies the
plan (`query_analysis`, `research_plan`), claims, citations, and - since Phase 9 - an
additive `evidence` list (`AnswerEvidence`: the passages given to the LLM, keyed by
`evidence_id`) so the UI can show the text behind each citation. `GET /health` and
`GET /retrieval/index/status` feed the status indicator.

Design decisions: claim status is taken from the verified top-level `claims` (section
claims are not verified by the backend and are rendered as "Not verified"); citations whose
evidence is missing render an explicit unavailable state; Markdown from the LLM is rendered
without raw HTML, images are dropped and links are restricted to http(s); the evidence panel
is a docked column on wide screens and a modal sheet (Escape to close) on narrow screens.

### 4.5 Runtime hardening (Phase 10)

* **Services**: `api/deps.py` holds one retrieval/planning/answer/indexing service per process; the cross-encoder is a shared, lock-protected singleton. Network clients are closed on shutdown.
* **Errors**: all handlers in `core/error_handling.py` return `{error, code, request_id}`; details go to redacted logs (`core/redaction.py`).
* **Outbound HTTP**: user-supplied URLs go through `SafeAsyncHttpClient(ssrf_protect=True, max_bytes=...)` (per-hop validation, IP pinning, streaming size cap).
* **Readiness**: `/ready` = required dependencies for answering (PostgreSQL, index, LLM config); `/health` = liveness.

## 5. Data Storage Architecture

### 5.1 PostgreSQL — Application Metadata

Stores structured metadata about research sessions, queries, and indexed documents. Does NOT store document content or vectors.

**Key Tables:**

| Table | Purpose |
|---|---|
| `research_sessions` | Groups related queries from a user session |
| `research_queries` | Each user query, its decomposed sub-queries, status, latency |
| `retrieved_documents` | Metadata of documents retrieved per query (URL, title, source) |
| `citations` | Mapping of citation markers in generated answers to documents |
| `indexed_documents` | Registry of documents ingested into OpenSearch + Qdrant |
| `evaluation_runs` | Records of offline evaluation runs and metric results |

### 5.2 OpenSearch — Lexical (BM25) Search

Stores the full text of ingested documents in an inverted index. Used for keyword and BM25 retrieval.

**Index design:**
- One primary index: `documents`
- Fields: `doc_id`, `title`, `content`, `source_type`, `source_url`, `published_date`, `authors`, `keywords`
- Analyzer: English analyzer with stemming
- BM25 is OpenSearch's default similarity — no additional configuration needed for baseline

### 5.3 Qdrant — Dense (Vector) Search

Stores document chunk embeddings for semantic similarity search.

**Collection design:**
- One primary collection: `document_chunks`
- Each point: embedding vector + payload (chunk_id, doc_id, text, source_url, title, chunk_index)
- Chunks are passage-level (not full documents) for more precise retrieval
- Vector dimension: 384 (default with `all-MiniLM-L6-v2`), configurable

### 5.4 Redis — Cache & Async Queue

Used in two roles:
1. **Celery broker/backend**: For async pipeline execution (longer research tasks)
2. **Query result cache**: Identical queries within a TTL window skip re-retrieval

Redis is optional in the minimal local setup and activates in production mode.

---

### 5.5 Canonical Data Root, Index Manifest & Generations (Phase 8.6)

All persisted retrieval state lives under ONE data root that never depends on the
working directory:

```
<data_root>/                      default: <repo>/data   (override: DATA_DIR; relative -> repo root)
  corpus/corpus.json              authoritative source documents for index builds
  evaluation/*.json               benchmark datasets (labels are never modified by indexing)
  indexes/
    ACTIVE                        {"generation_id": ...} pointer, replaced atomically
    gen-<UTC>-<hex>/
      manifest.json               embedding + chunking + counts manifest
      bm25_index.json             local BM25 store (lexical only, tagged with the generation)
      qdrant/                     embedded Qdrant, collection `document_chunks`
  backups/<UTC>_<label>/          sha256-manifested snapshots taken before migrations
```

**Manifest** (`indexes/<gen>/manifest.json`): `schema_version`, `generation_id`, `status`
(`building`|`complete`), `created_at`, `completed_at`, `embedding{provider, model_name,
dimension, revision, normalized, is_mock}`, `chunking{chunk_size_tokens,
chunk_overlap_tokens, chunker_version}`, `counts{documents, chunks}`, `collection_name`,
`corpus{sha256, ...}`. Every Qdrant payload also carries `index_generation` and
`embedding_model`.

**Fail-closed rules** (all enforced in `retrieval/index_manifest.py`):
* Missing, unreadable, incomplete (`building`) or old-schema manifest -> request fails (HTTP 503, code `index_manifest_missing` / `index_build_incomplete` / `index_stale`).
* Model identity is compared, not just dimension: provider, model name, normalization, mock-vs-real and (when both known) model revision. A 384-d mock vector is never accepted for `all-MiniLM-L6-v2` (`embedding_model_mismatch`).
* Results whose payload generation differs from the active manifest are rejected (`index_generation_mismatch`).
* Writers additionally reject a changed chunking configuration (`index_stale`) and validate every vector (dimension, finite, non-zero, unit norm).
* The mock embedder exists only when `EMBEDDING_PROVIDER=mock` is configured explicitly. If the real model cannot load, `EmbeddingModelLoadError` (503) is raised; nothing is substituted or written.
* Dense/hybrid retrieval never degrades silently to BM25 when the dense index is untrustworthy.

**Duplicate-safe writes**: document ids are `uuid5(source_type:source_id)` and chunk ids
`uuid5(document_id:chunk_index)`. Re-indexing replaces a document's chunks in both stores.

**Rebuild procedure** (`scripts/index_admin.py`, works from any directory):

```powershell
python scripts/index_admin.py inventory      # read-only: legacy stores + generations
python scripts/index_admin.py backup         # snapshot legacy stores into <data>/backups/
python scripts/index_admin.py build-corpus   # (re)assemble corpus/corpus.json
python scripts/index_admin.py rebuild        # validate provider -> build NEW generation -> verify -> activate
python scripts/index_admin.py verify         # deep integrity check of the active generation
python scripts/index_admin.py status         # manifest summary + compatibility with configured provider
python scripts/index_admin.py generations    # list generations
python scripts/index_admin.py prune          # delete generations that never completed
python scripts/verify_retrieval_paths.py     # independent BM25 / dense / hybrid / rerank verification
```

**Recovery**: a rebuild writes into a fresh generation directory and activates it only after
`verify_integrity` passes. If it is interrupted or fails, the previous generation stays
active and the partial one is flagged `building` (never served; remove with `prune`).
`GET /api/v1/retrieval/index/status` reports the active generation, embedding identity and
compatibility without exposing filesystem paths.

## 6. Retrieval Pipeline Design

### 6.1 Indexing Flow (Document Ingestion)

```
Source (arXiv / web / manual) 
    → Fetch raw content
    → Clean & extract text
    → Chunk into passages (512 tokens, 50-token overlap)
    → Index full text into OpenSearch
    → Embed each chunk via EmbeddingProvider
    → Upsert embedding into Qdrant
    → Register metadata in PostgreSQL
```

### 6.2 Query-Time Retrieval Flow

**Step 1 — BM25 Retrieval**
- Query → OpenSearch `multi_match` on `title`, `content`, `keywords`
- Return top-N BM25 candidates (N = 30–50 configurable)
- Each result includes: BM25 score, doc_id, title, snippet

**Step 2 — Dense Retrieval**
- Query → embed with EmbeddingProvider → query vector
- Qdrant ANN search → top-N semantic neighbors (N = 30–50)
- Each result includes: cosine similarity score, chunk_id, doc_id

**Step 3 — Hybrid Fusion (Reciprocal Rank Fusion)**

RRF is chosen over simple score interpolation because:
- BM25 and dense scores are not on the same scale
- RRF is robust to this: it uses rank positions, not raw scores
- Formula: `RRF_score(d) = Σ 1 / (k + rank_i(d))` where `k = 60` (standard)
- Produces a unified ranked list of candidates

**Step 4 — Deduplication**

Documents appearing in both BM25 and dense results are merged (the better rank is kept). The merged list is deduplicated by `doc_id`.

**Step 5 — Cross-Encoder Reranking**

- Input: (query, passage) pairs for top-M candidates (M = 20–30)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (default)
- Output: relevance score per (query, passage) pair
- Reranked list sorted descending by cross-encoder score
- Return top-K (K = 5–10 configurable per request)

**Why cross-encoders vs bi-encoders for reranking:**
Cross-encoders jointly encode the query and document together, capturing fine-grained relevance signals that bi-encoders (used for dense retrieval) miss because they encode independently. The tradeoff is speed — cross-encoders are slower but are only applied to the already-filtered top-M candidates.

### 6.3 Source Connectors

Each connector implements the `DocumentSource` protocol:

```python
class DocumentSource(Protocol):
    async def search(self, query: str, max_results: int) -> list[SourceDocument]:
        ...
    async def fetch(self, identifier: str) -> SourceDocument:
        ...
```

| Connector | Source | What it retrieves |
|---|---|---|
| `ArxivConnector` | arXiv API | Paper abstracts, full PDF text (via arxiv2text) |
| `SemanticScholarConnector` | Semantic Scholar API | Paper metadata, abstracts, citations |
| `WebFetcher` | Any URL / search engine | Cleaned article text via trafilatura |
| `GitHubConnector` (future) | GitHub API | README content, code comments, issues |
| `LocalDocsConnector` (future) | Local filesystem | User-uploaded PDFs, Markdown |

---

## 7. AI / ML Components

### 7.1 Embedding Provider

**Interface:** `EmbeddingProvider`

```python
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...
    @property
    def dimension(self) -> int:
        ...
```

**Default implementation:** `SentenceTransformerProvider` using `all-MiniLM-L6-v2` (lazy, fail-closed loading; identity incl. model revision recorded in the index manifest). `MockEmbeddingProvider` is selectable only via `EMBEDDING_PROVIDER=mock`.
- 384-dimensional embeddings
- Fast on CPU, no GPU required for moderate-scale indexing
- Runs entirely locally — no API cost

**Future options (config-swappable):**
- `text-embedding-3-small` (OpenAI) — higher quality, API cost
- `e5-large-v2` (local) — higher quality, higher compute
- `nomic-embed-text` (local Ollama) — competitive, free

### 7.2 Reranker

**Interface:** `Reranker`

```python
class Reranker(Protocol):
    def rerank(self, query: str, documents: list[Document]) -> list[RankedDocument]:
        ...
```

**Default implementation:** `CrossEncoderReranker` using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- State-of-the-art passage reranking
- Runs locally on CPU (inference on 20 passages is fast)
- Trained on MS MARCO — strong general relevance model

**Future options:**
- `cross-encoder/ms-marco-electra-base` — higher quality, slower
- `Cohere Rerank API` — API-based, very high quality

### 7.3 LLM Provider

See Section 9 for full detail. Default: Ollama (local, zero-cost) for development, OpenAI or Anthropic for production.

### 7.4 Query Intelligence

Uses the configured LLM with **structured output** (Pydantic models via instructor or OpenAI function calling) to produce:

```python
class QueryAnalysis(BaseModel):
    intent: Literal["research", "factual", "comparison", "tutorial"]
    key_concepts: list[str]
    sub_queries: list[str]          # 3–7 targeted queries
    search_dimensions: list[str]    # e.g. ["accuracy", "latency", "hardware"]
    suggested_sources: list[str]    # e.g. ["arxiv", "web"]
```

This structured decomposition drives the retrieval strategy dynamically.

---

## 8. External Integrations

| Integration | Purpose | API Type | Rate Limits | Required? |
|---|---|---|---|---|
| **arXiv** | Research paper retrieval | REST (free) | ~3 req/sec | Optional |
| **Semantic Scholar** | Paper metadata + citations | REST (free tier) | 100 req/5min (unauth) | Optional |
| **OpenAI** | LLM synthesis, embeddings | REST (paid) | Varies by tier | Optional (dev uses Ollama) |
| **Anthropic Claude** | LLM synthesis alternative | REST (paid) | Varies by tier | Optional |
| **Ollama** | Local LLM inference | REST (local) | No limits | Optional (recommended for dev) |
| **GitHub** | Code/README search | REST (free tier) | 60 req/hour (unauth) | Optional |

All integrations are:
- Behind the `DocumentSource` or `LLMProvider` protocol
- Independently toggleable via config
- Wrapped with retry logic and timeout handling
- Designed to fail gracefully (pipeline continues without the unavailable source)

---

## 9. LLM Provider Abstraction

The system must not be tightly coupled to any single LLM provider. The abstraction:

```python
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        ...
```

**Why not LangChain for this?**
LangChain's abstractions add complexity and make debugging difficult. For a system that needs transparent, measurable pipeline stages, direct provider clients with a thin protocol layer are preferable. We use `instructor` for structured output (which wraps provider clients cleanly).

**Provider resolution:** selected via `LLM_PROVIDER` env var (`openai`, `anthropic`, `ollama`).

**Prompt design principles:**
- System prompt: instructs the LLM to answer ONLY from the provided context
- User prompt: includes numbered evidence passages with source attribution
- The LLM is instructed to use `[1]`, `[2]`, etc. citation markers in its response
- Post-processing maps markers back to real `SourceDocument` metadata

---

## 10. Evaluation Strategy

### 10.1 Retrieval Evaluation

Implemented in `backend/evaluation/metrics.py` and `evaluation/scripts/`.

**Metrics implemented:**

| Metric | Formula | Measures |
|---|---|---|
| **Recall@K** | |relevant ∩ retrieved@K| / |relevant| | Did we find the right documents? |
| **Precision@K** | |relevant ∩ retrieved@K| / K | How many of top-K are relevant? |
| **MRR** | 1/rank of first relevant result | How early does the first relevant result appear? |
| **NDCG@K** | Normalized Discounted Cumulative Gain | Quality of the entire ranking, rewards relevant results at top positions |

**Evaluation dataset format:**
```json
{
  "query": "YOLO vs transformer object detectors edge deployment",
  "relevant_doc_ids": ["arxiv:2304.00123", "arxiv:2210.45678", "semscholar:xyz"]
}
```

**How to run:**
```bash
python evaluation/scripts/run_eval.py \
    --dataset evaluation/datasets/retrieval_test_v1.json \
    --k 5 10 20 \
    --retriever hybrid
```

### 10.1 Evaluation Strategy & Multi-Variant Benchmark (Phase 7)

Phase 7 implements a formal, reproducible scientific evaluation suite comparing 6 architectural variants:

```text
Evaluation Dataset
        ↓
     Question
        ↓
┌─────────────────────────────────────────┐
│             SYSTEM VARIANTS             │
├─────────────────────────────────────────┤
│ A. BM25 (Lexical Only)                  │
│ B. Dense Retrieval (Vector Only)        │
│ C. Hybrid Retrieval (BM25 + Dense RRF)  │
│ D. Hybrid + Cross-Encoder Reranking     │
│ E. Query Decomposition + Hybrid         │
│ F. Full Research Pipeline (End-to-End)  │
└─────────────────────────────────────────┘
        ↓
Retrieval Metrics (Recall@K, Precision@K, MRR, NDCG@K)
        ↓
Answer Grounding & Verification (Claim Support, Citation Validity)
        ↓
Granular Latency Profiling & Automated Ablation Reports
```

### 10.2 Information Retrieval & Answer Metrics

- **Recall@K & Precision@K**: Proportion of unique relevant documents retrieved at $K \in \{1, 5, 10\}$.
- **Mean Reciprocal Rank (MRR)**: Reciprocal rank ($1/\text{rank}$) of the first retrieved relevant document.
- **NDCG@K**: Normalized Discounted Cumulative Gain with binary relevance, strictly bounded in $[0.0, 1.0]$.
- **Evidence Dimension Coverage**: Proportion of expected technical dimensions captured in retrieved text.
- **Claim Support Rate**: Fraction of factual claims verified as fully supported by retrieved evidence.
- **Citation Validity & Traceability**: Verification of citations linking claims to passage IDs, document IDs, and source URLs.

### 10.3 Execution & Reproduction

```powershell
# Full benchmark evaluation
python scripts/evaluate_system.py

# Retrieval ablation only (Variants A–E)
python scripts/evaluate_system.py --retrieval-only

# Answer generation & verification (Variant F)
python scripts/evaluate_system.py --answers
```

---

## 11. Technology Decisions & Rationale

### 11.1 OpenSearch vs Elasticsearch

**Decision: OpenSearch**

OpenSearch is a fully open-source fork of Elasticsearch 7.10 maintained by AWS. It has no licensing restrictions for self-hosting. The API is nearly identical to Elasticsearch for our use cases (BM25, `multi_match`, index management). Since we are self-hosting via Docker Compose and not using cloud-managed Elasticsearch, OpenSearch is the correct choice.

**If this changes:** Switching to Elasticsearch would require minimal code changes — only the client initialization and any OpenSearch-specific plugin calls.

### 11.2 Qdrant vs Pinecone vs Weaviate vs pgvector

**Decision: Qdrant**

| Option | Reason Not Chosen |
|---|---|
| **Pinecone** | Managed-only, no local Docker option, API cost |
| **Weaviate** | Heavier resource footprint, more complex configuration |
| **pgvector** | Good for small-scale but lacks ANN efficiency at scale, no dedicated vector index management |
| **ChromaDB** | Appropriate for simple RAG demos, not production-grade at scale |

Qdrant runs locally via Docker, has excellent Python client, supports payload filtering, and has strong performance benchmarks. It is the correct choice for a production-quality local-first system.

### 11.3 Why Reciprocal Rank Fusion (RRF) over Score Interpolation

BM25 scores and cosine similarity scores are on entirely different scales. Linear interpolation (`alpha * bm25 + (1-alpha) * dense`) requires careful alpha tuning and is sensitive to score magnitude. RRF uses only rank positions:

```
RRF(d) = Σ_r 1 / (k + r(d))
```

Where `k=60` is a smoothing constant. This is robust, parameter-free, and empirically competitive with tuned interpolation. It is the standard approach in the literature.

### 11.4 Why `sentence-transformers` as Default Embedder

- Runs entirely locally on CPU
- No API key required for development
- `all-MiniLM-L6-v2` is fast and well-benchmarked
- The `EmbeddingProvider` abstraction means we can swap to OpenAI or a larger model trivially

### 11.5 Why not LangChain as the core framework

LangChain is excellent for rapid prototyping. For a production research system:
- LangChain chains are opaque — debugging a retrieval failure requires understanding LangChain internals
- Our pipeline stages need to be individually measurable and replaceable
- We want explicit data contracts between stages (Pydantic models), not implicit chain passing
- LangChain adds significant dependency weight

We use `instructor` for structured LLM output (it wraps provider clients cleanly without framework lock-in), but the pipeline logic is explicit Python.

### 11.6 Why FastAPI over Flask or Django

- Native async support matches our async I/O pattern (OpenSearch, Qdrant, LLM calls all benefit from async)
- Pydantic integration is first-class
- Auto-generated OpenAPI docs
- High performance

### 11.7 Why Celery + Redis for async tasks

Long research pipelines (multiple sub-queries, multiple source fetches) may take 10–30 seconds. Celery allows:
- Non-blocking API responses (return a job ID, poll for results)
- Pipeline retries on transient failures
- Future: parallel sub-query execution

Redis is already in the stack as a Celery broker; it doubles as a query result cache with minimal additional complexity.

---

## 12. Repository Structure

> **Planned layout (Phase 0), kept as a design record.** `docker-compose.yml`, `services/`, `infrastructure/` and the root-level `evaluation/` directory shown below were never populated and have been removed; benchmark data lives in `data/evaluation/`, evaluation code in `backend/evaluation/` and `scripts/`, and reports in `reports/evaluation/`. See the root [README](../README.md#project-structure) for the actual layout.

```
ai-research-copilot/
│
├── README.md                          # Project overview and setup guide
│
├── .env.example                       # Template for all environment variables
├── .gitignore                         # Excludes .env, __pycache__, node_modules, etc.
│
├── docker-compose.yml                 # Local orchestration: all services
│
├── frontend/                          # React + TypeScript + Vite
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── store/
│   │   └── types/
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── package.json
│
├── backend/                           # FastAPI Python application
│   ├── main.py
│   ├── requirements.txt
│   ├── core/
│   ├── api/
│   ├── services/
│   ├── retrieval/
│   ├── ranking/
│   ├── synthesis/
│   ├── connectors/
│   ├── embedding/
│   ├── llm/
│   ├── db/
│   └── evaluation/
│
├── services/                          # Service-specific config files
│   ├── opensearch/
│   │   └── opensearch.yml             # Custom OpenSearch config
│   └── qdrant/
│       └── config.yaml                # Qdrant collection config
│
├── data/
│   ├── test_corpus/                   # Small document set for dev testing
│   └── seed_queries.json              # Sample research queries for testing
│
├── evaluation/
│   ├── datasets/                      # Evaluation query + relevance datasets
│   ├── scripts/
│   │   ├── run_eval.py
│   │   └── generate_queries.py
│   └── results/                       # Stored evaluation run outputs
│
├── infrastructure/
│   ├── docker/
│   │   ├── backend.Dockerfile
│   │   └── frontend.Dockerfile
│   └── scripts/
│       └── wait-for-it.sh             # Service readiness helper
│
├── docs/
│   ├── ARCHITECTURE.md                # This document
│   ├── adr/                           # Architecture Decision Records
│   │   └── 001-technology-choices.md
│   └── api/                           # API documentation (auto-generated + manual)
│
└── scripts/
    ├── setup_local.sh                 # Full local dev setup script
    ├── seed_opensearch.py             # Index test corpus into OpenSearch
    └── seed_qdrant.py                 # Index test corpus into Qdrant
```

---

## 13. Phased Implementation Roadmap

### Phase 0 — Architecture & Scaffolding ✅ (Current)
**Goal:** Define architecture, create repo structure, document decisions.

Deliverables:
- `docs/ARCHITECTURE.md`
- `README.md`
- `.env.example`
- `.gitignore`
- `docker-compose.yml` skeleton
- Empty directory structure

---

### Phase 1 — Backend Foundation
**Goal:** A running FastAPI application with health endpoints, config system, and DB models.

Deliverables:
- `backend/main.py` with FastAPI app
- `backend/core/config.py` (pydantic-settings, reads .env)
- `backend/core/logging.py` (structlog structured logging)
- `backend/api/v1/health.py` (GET /health, GET /ready)
- SQLAlchemy async database setup
- ORM models for `ResearchQuery`, `IndexedDocument`, `ResearchSession`
- Alembic migration setup
- `requirements.txt`

---

### Phase 2 — Retrieval Core
**Goal:** Working hybrid retrieval against OpenSearch + Qdrant.

Deliverables:
- `backend/embedding/sentence_transformer_provider.py`
- `backend/retrieval/bm25_retriever.py`
- `backend/retrieval/dense_retriever.py`
- `backend/retrieval/hybrid_retriever.py` (RRF)
- `backend/retrieval/passage_chunker.py`
- `scripts/seed_opensearch.py` and `scripts/seed_qdrant.py`
- Docker Compose updated with OpenSearch + Qdrant services
- Integration tests for each retriever

---

### Phase 3 — Source Connectors
**Goal:** Fetch real documents from arXiv and Semantic Scholar.

Deliverables:
- `backend/connectors/arxiv_connector.py`
- `backend/connectors/semantic_scholar_connector.py`
- `backend/connectors/web_fetcher.py`
- Connector retry logic and rate limiting
- Ingestion pipeline: fetch → chunk → index

---

### Phase 4 — Reranking & Evidence Extraction
**Goal:** Cross-encoder reranking of retrieved candidates.

Deliverables:
- `backend/ranking/cross_encoder_reranker.py`
- Evidence extraction logic (best passage per document)
- Reranker evaluation against BM25-only baseline

---

### Phase 5 — LLM Synthesis & Citations
**Goal:** Generate grounded answers from retrieved evidence with traceable citations.

Deliverables:
- `backend/llm/base.py` + provider implementations (OpenAI, Anthropic, Ollama)
- `backend/synthesis/prompt_builder.py`
- `backend/synthesis/answer_generator.py`
- `backend/synthesis/citation_engine.py`
- `/api/v1/research/query` endpoint (full pipeline end-to-end)

---

### Phase 6 — Query Intelligence & Research Decomposition [COMPLETED]
**Goal:** Decompose complex research queries into targeted subqueries, execute parallel retrieval, and merge evidence with deterministic coverage metrics.

Deliverables:
- `backend/query/models.py` (Typed Pydantic models for intent, complexity, subqueries, plan, merged evidence)
- `backend/query/complexity.py` (Deterministic complexity detection with simple-query optimization)
- `backend/query/intent.py` (Layered intent classifier: deterministic rules + LLM fallback)
- `backend/query/decomposer.py` (QueryDecomposer for rule-based and LLM-assisted decomposition)
- `backend/query/merger.py` (EvidenceMerger for deduplication, diversity budgeting, provenance tracking, coverage metrics)
- `backend/query/planner.py` (ResearchPlanningService orchestrating parallel retrieval via asyncio.gather)
- `backend/api/v1/research.py` (Added `POST /api/v1/research/plan` and updated `POST /api/v1/research/answer`)
- `data/query_decomposition_benchmark.json` & `backend/evaluation/decomposition_evaluator.py` (Benchmark evaluation dataset and suite)
- `docs/QUERY_INTELLIGENCE.md` (Architecture and evaluation documentation)


---

### Phase 7 — Frontend
**Goal:** A research-quality UI that shows answers, citations, and process trace.

Deliverables:
- React + Vite + TypeScript project setup
- Search bar with query input
- Results panel with cited answer
- Citations list with source metadata
- Process trace panel (sub-queries, sources used)
- Comparison mode (for side-by-side technology comparisons)

---

### Phase 8 — Evaluation Framework
**Goal:** Measurable retrieval quality with standard IR metrics.

Deliverables:
- `backend/evaluation/metrics.py` (Recall@K, Precision@K, MRR, NDCG)
- Evaluation dataset (at least 20 queries with ground-truth relevance)
- `evaluation/scripts/run_eval.py`
- Latency breakdown in API responses
- Baseline evaluation: BM25-only vs hybrid vs hybrid+rerank

---

### Phase 9 — Hardening & Observability
**Goal:** Production-ready error handling, rate limiting, basic auth, and observability.

Deliverables:
- Request rate limiting
- API key auth (optional header)
- Structured logging across all pipeline stages
- Prometheus metrics endpoint
- Graceful degradation when external sources are unavailable
- Docker Compose production profile

---

## 14. Security & Configuration

### 14.1 Environment Variables

All sensitive configuration is loaded via `pydantic-settings` from environment variables. See `.env.example` for the complete list.

Never commit `.env` to version control.

### 14.2 Secret Categories

| Variable Group | Description |
|---|---|
| `LLM_*` | LLM provider selection and API keys |
| `OPENAI_*` | OpenAI API credentials |
| `ANTHROPIC_*` | Anthropic API credentials |
| `OPENSEARCH_*` | OpenSearch host, port, credentials |
| `QDRANT_*` | Qdrant host, port, API key (if cloud) |
| `POSTGRES_*` | PostgreSQL connection details |
| `REDIS_*` | Redis connection details |
| `SEMANTIC_SCHOLAR_*` | API key for higher rate limits |
| `GITHUB_*` | Personal Access Token for GitHub connector |

### 14.3 No Secrets in Code

The codebase must never contain:
- API keys or tokens
- Database passwords
- Any credential, even for testing

Use environment variables exclusively, with `os.getenv()` or `pydantic-settings`.

---

## 15. MANUAL ACTION REQUIRED

The following items require manual action from the developer and cannot be automated. They are NOT needed during Phase 0 (architecture), but will be needed in later phases.

---

### [MA-001] LLM Provider API Key

**When needed:** Phase 5 (LLM Synthesis)

**Options:**

**Option A — Local Ollama (Recommended for development, free)**
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2` or `ollama pull mistral`
3. Set in `.env`: `LLM_PROVIDER=ollama` and `OLLAMA_BASE_URL=http://localhost:11434`

**Option B — OpenAI (Production quality)**
1. Create account at https://platform.openai.com
2. Generate API key at https://platform.openai.com/api-keys
3. Set in `.env`: `LLM_PROVIDER=openai` and `OPENAI_API_KEY=sk-...`

**Option C — Anthropic Claude**
1. Create account at https://console.anthropic.com
2. Generate API key
3. Set in `.env`: `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=sk-ant-...`

---

### [MA-002] Semantic Scholar API Key (Optional, Higher Rate Limits)

**When needed:** Phase 3 (Source Connectors)

The Semantic Scholar API works without a key but is rate-limited to 100 requests per 5 minutes. For production use:
1. Request an API key at: https://www.semanticscholar.org/product/api
2. Set in `.env`: `SEMANTIC_SCHOLAR_API_KEY=your-key`

---

### [MA-003] GitHub Personal Access Token (Optional, GitHub Connector)

**When needed:** Phase 3+ (when GitHub connector is enabled)

1. Go to: https://github.com/settings/tokens
2. Generate a fine-grained token with `public_repo` read access
3. Set in `.env`: `GITHUB_TOKEN=ghp_...`

---

### [MA-004] arXiv API — No Key Required

arXiv's public API does not require authentication. Rate limit is approximately 3 requests per second. No action needed.

---

### [MA-005] PostgreSQL Credentials

**When needed:** Phase 1 (Backend Foundation)

For local development, the Docker Compose file will create a PostgreSQL instance with default credentials defined in `.env`. You only need to:
1. Copy `.env.example` to `.env`
2. Choose secure values for `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`

For production deployment, use a managed database (AWS RDS, Supabase, etc.) and update the connection string.

---

*End of ARCHITECTURE.md*

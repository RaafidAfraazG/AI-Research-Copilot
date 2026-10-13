# AI Research Copilot — Claude Code Handover & Audit

**Audit date:** 2026-10-10 · **Auditor:** Claude Code (Sonnet 5.5) · **Scope:** audit only, no code/env/data changed.
**Source of truth:** the repository root (the directory containing this `docs/` folder). Not a git repository (no history, no VCS safety net — see §14).
**Labels used:** VERIFIED · PARTIALLY VERIFIED · IMPLEMENTED, NOT RUN · MISSING · BROKEN. Anything I could not check is marked **UNVERIFIED**.

---

## 0. TL;DR

* The backend is a real, well-structured retrieval → rerank → evidence → grounded-synthesis → citation-check pipeline. 239 offline tests pass.
* **Since the last summary, real models were validated** (2026-10-01): the neural cross-encoder runs, and Gemini free tier answered real requests. The previous "NOT VALIDATED" statements are partly stale.
* **But the evaluation numbers in README/reports are stale and not trustworthy**: they were produced on 2026-09-29 with a *mock hash embedder*, a *fallback (no-op) reranker*, and a *mock LLM*, on a **corpus of 8 documents**.
* **_[superseded 8.6 — repaired, see §0.1]_ Critical latent defect:** the persisted vector index (`data/qdrant_local`, and the `embedding` fields in `data/opensearch_local/bm25_index.json`) holds **mock hash vectors**, but `sentence-transformers` is now installed, so queries are embedded by the **real** MiniLM model. Same dimension (384), so nothing errors; dense retrieval silently compares vectors from two unrelated spaces. Measured cosine(stored, real) for the same text = **0.03**.
* **Citation "verification" is structural, not semantic.** A claim is `SUPPORTED` iff its `evidence_ids` exist in the evidence pack. Nothing checks that the passage actually entails the claim. "Claim support rate 1.0" proves ID validity only.
* There is **no frontend** (empty `frontend/src/*` dirs). The product is API-only.

---

## 0.1 Phase 8.6 addendum (2026-10-10) — index integrity repaired

**This addendum supersedes the audit statements it contradicts below (marked _[superseded 8.6]_).**

| Audit finding | State after Phase 8.6 |
|---|---|
| Live index held mock hash vectors, queried with the real embedder (cosine 0.03) | **Fixed.** New generation `gen-20261010T062446Z-34c4e8` built entirely with real `all-MiniLM-L6-v2` (rev `1110a243…`, 384-d, normalized). Vectors were generated, not copied. |
| No embedding identity/dimension validation | **Fixed.** `retrieval/index_manifest.py`: manifest + identity validation (provider, model, dimension, normalization, mock flag, revision), generation tags on every vector, fail-closed errors (HTTP 503). |
| Silent fallback to mock embeddings | **Fixed.** `EMBEDDING_PROVIDER=mock` is the only way to get mock vectors. Real-model load failure raises `EmbeddingModelLoadError`. |
| Two divergent, cwd-relative stores | **Fixed.** Single data root `core/paths.py::resolve_data_root` (`<repo>/data` or `DATA_DIR`). Legacy stores are no longer read; they remain on disk and in `data/backups/`. |
| Duplicate chunks / test pollution in the index | **Fixed.** Deterministic uuid5 document/chunk ids with replace-on-reindex; test suite runs in a per-test temp `DATA_DIR`. Legacy index had 56 chunks for 8 documents (24 for one 2-chunk paper, 22 copies of a test fixture). |
| Dense error swallowed into empty results; planner hid index errors as "failed subquery" | **Fixed** for integrity/embedding errors (propagate as 503). Other transient dense errors still return an empty result with `metadata.error`. |

**Corpus & counts**

| | Legacy root `data/` | Legacy `backend/data/` | Rebuilt canonical index |
|---|---|---|---|
| Documents | 8 | 2 | **7** |
| Chunks | 56 | 10 (8 at audit; +2 from the audit's own test run) | **7** |
| Vectors | mock hash (384-d / 64-d mix) | mock | **real MiniLM, 384-d** |

Corpus `data/corpus/corpus.json` = 5 papers from `data/sample_papers.json` + `2608.09650` (real arXiv
abstract, recovered verbatim from its single legacy chunk) + `test.integration.001` (test fixture;
recovered because one benchmark label references it — labels were NOT changed). Excluded as pollution:
`test-doc-1` and an unit-test string that had been written into `2005.11401`. Backups with sha256
manifests: `data/backups/20261010T062048Z_legacy_root_data` and `…_legacy_backend_data` (verified identical
to the sources before the rebuild). Nothing was deleted. An earlier same-day generation
`gen-20261010T062209Z-fcfe85` (no revision recorded) remains for reference; `generations`/`prune` manage it.

**Verified paths** (`scripts/verify_retrieval_paths.py`, report `reports/evaluation/phase86_retrieval_verification.json`): BM25 PASS;
dense PASS (scores equal cosine(stored vector, freshly embedded query), max |err| 6.4e-8); hybrid PASS;
neural cross-encoder PASS (loaded, not fallback, scores are logits); mock rejected by real index PASS.

**Tests:** 267 passed, 4 skipped (opt-in real-model tests), 36 warnings; with `RUN_REAL_MODEL_TESTS=1` the 4 real-model tests pass.
New: `backend/tests/test_index_integrity.py` (28), `backend/tests/test_real_embeddings.py` (4, opt-in).

**Benchmark (variants A–D, real models):** all variants saturate (R@5 = R@10 = 1.0) on a 7-chunk corpus;
MRR 1.000 / 0.973 / 1.000 / 1.000 for BM25 / dense / hybrid / hybrid+rerank. This does **not** show that any
method is better; see README "Evaluation". Old mock-based reports are archived in
`reports/evaluation/historical_mock_phase7/` with a NOTICE.

**New operational interface:** `scripts/index_admin.py {inventory,backup,build-corpus,rebuild,verify,status,generations,prune}`,
`GET /api/v1/retrieval/index/status`, `DATA_DIR`, `EMBEDDING_PROVIDER=local|openai|mock`.

**Still open after 8.6:** citation verification is structural only; reranker/retrievers are rebuilt per HTTP request
(cold model load); `/ready` still demands OpenSearch+Redis; SSRF redirect re-validation; no auth; Gemini end-to-end
through the real pipeline not yet validated (smoke-test script field bug); E/F variants not re-run with real models;
corpus far too small for conclusions; remote OpenSearch/Qdrant write paths are not covered by the manifest (not used locally).

## 0.2 Phase 9 addendum — frontend

* `frontend/` (React + TS + Vite + Tailwind) implemented; see README "Quick start" and ARCHITECTURE section 4.4.
* Single backend change: `ResearchAnswer.evidence: list[AnswerEvidence]` (additive, default `[]`), filled in
  `synthesis/answer_service.py` from the prompt's id->passage map. Needed because citations carried no passage text.
* Integrated endpoints: `GET /api/v1/health`, `GET /api/v1/retrieval/index/status`, `POST /api/v1/research/answer`.
* Backend quirk handled in the UI: claims inside `sections` are not run through the verifier (their `support_status` is
  the model default). The UI takes status from the verified top-level `claims` by text match and otherwise shows "Not verified".
* Not available: session history (no list/save API), structured LLM quota errors (500 `detail` text is pattern-matched
  for rate limits and never displayed), pipeline-stage progress (single blocking request).
* Validation in this phase was deliberately light: type-check, production build, 9 unit tests, and one manual smoke run
  (backend with `LLM_PROVIDER=mock`, UI -> answer -> citation panel). No real-Gemini UI run, no full backend suite.

## 0.3 Phase 10 addendum — integration hardening

| Defect | Fix |
|---|---|
| Gemini validation script used non-existent verification fields (`citation_validity_rate`, ...) | `scripts/phase85_smoke_test.py` fields corrected. New bounded `scripts/gemini_e2e_validate.py`: ONE Gemini request, no retries, configured model, no key output, exit 2 without key. **Real run PASSED 2026-10-10** (see report `reports/evaluation/gemini_e2e_validation.json`). |
| Truncated/malformed LLM JSON became an "insufficient evidence" answer (found by the real run) | `answer_service` now raises `LLMGenerationError`; API returns 502 `llm_malformed_output`. |
| Cross-encoder (and retriever clients) constructed per request | `ranking_service.get_shared_reranker()` process singleton (thread-safe load + serialized inference, 60 s retry cooldown after a failed load); `api/deps.py` shared services; closed in lifespan via `aclose_services()`. |
| SSRF: redirects not re-validated, size checked after full buffering | `connectors/url_security.py` + hardened `SafeAsyncHttpClient.get(ssrf_protect=, max_bytes=)`: every hop validated (scheme, credentials, all resolved IPs public, IPv4-mapped/6to4/Teredo unwrapped), connects to the validated IP with Host/SNI, streaming byte cap (decoded bytes), declared Content-Length check, redirect cap, connect + total deadlines. DNS rebinding is narrowed, not eliminated (proxies via env are ignored in this mode). |
| Raw exception text in API responses / logs | `core/error_handling.py` stable `{error, code, request_id}` bodies; `core/redaction.py` redacts secrets (configured values + patterns) and normalises paths in all logs and tracebacks. |
| `/ready` 503 because OpenSearch/Redis are offline | Required: PostgreSQL, index integrity + embedding-config match, LLM configuration (no provider call). OpenSearch/Qdrant-server/Redis: `skipped` unless `READINESS_PROBE_OPTIONAL=true`, then informational. `/health` stays pure liveness. |
| Section claims bypassed verification | Section claims are the same verified objects as top-level claims; section-only claims are appended and verified; sections materialised after verification. `CitationVerificationResult` now exposes `verification_method="structural_evidence_id_check"` and `semantic_entailment_checked=false`. |

Remaining after Phase 10: semantic entailment is still not implemented (status = structural only); no auth/rate limiting;
`/research/*` is not request-concurrency-limited (one reranker lock serializes inference); remote OpenSearch/Qdrant write
paths are outside the manifest; the benchmark corpus is still 7 documents; the legacy 8.5 smoke script still contains
multi-call retry/wait logic and should not be used for Free Tier validation (use `gemini_e2e_validate.py`).

## 0.4 Phase 11 addendum — final validation

Status **PASS WITH LIMITATIONS**; full report `reports/evaluation/phase11_final_validation.md`. Backend 325 passed / 4 opt-in skipped;
frontend build + 10 tests pass; real Gemini end-to-end passed (1 request); corpus extended to 41 documents (active generation
`gen-20261010T080354Z-16e621`, older generations retained); benchmark near-saturated, no method ranking claimed. Open: structural-only
citation verification, 58 s cold first request (no warm-up), no auth/rate limiting, tiny/incompletely labeled corpus.

## 0.5 Phase 12 addendum — portfolio release

README rewritten (architecture diagram, setup, results, limitations); `docs/DEMO.md` and `docs/PORTFOLIO.md` added; stale docs carry status
notes; `.gitignore` covers local data/indexes/backups; absolute `file:///` links removed. One code change: the first-use embedding model
load in `DenseRetriever.search` now runs via `asyncio.to_thread` (it previously froze the event loop for ~50 s on a cold start; total cold
latency is unchanged). See `reports/evaluation/phase12_portfolio_release.md`.

---

## 1. Project overview and goals

A research-intelligence backend, intentionally not a PDF-upload chatbot. Target pipeline: multi-source ingestion (arXiv, Semantic Scholar, web) → chunking → BM25 + dense indexes → hybrid RRF retrieval → cross-encoder rerank → evidence selection → LLM synthesis with claim-level citations → deterministic citation verification → benchmarked evaluation. Constraints (no Docker/Redis/OpenSearch daemon requirement, no paid APIs, Gemini free tier + MockLLM) are respected by the current code; see §10.

## 2. Current architecture

```
POST /api/v1/research/answer
  └─ ResearchAnswerService (synthesis/answer_service.py)
       ├─ enable_decomposition=true  → ResearchPlanningService.plan_and_retrieve (query/planner.py)
       │     QueryDecomposer (intent, complexity, subqueries; LLM or deterministic)
       │     → parallel ResearchRetrievalService.retrieve_evidence per subquery
       │     → EvidenceMerger (query/merger.py) → MergedEvidencePack
       └─ enable_decomposition=false → ResearchRetrievalService.retrieve_evidence
             HybridRetriever (BM25Retriever ∥ DenseRetriever → RRF/RSF)
             → RankingService → CrossEncoderReranker
             → EvidenceSelector (min-score, per-doc quota, Jaccard dedupe) → EvidencePack
       → GroundedPromptBuilder → LLMProvider.generate(response_format="json")
       → parse JSON → CitationMapper → CitationVerifier → ResearchAnswer
```

Storage (all actually used locally): PostgreSQL (port **5433**, app metadata only), a JSON-file BM25 store (`LocalBM25Store`), embedded Qdrant (`QdrantClient(path=...)`). Remote OpenSearch/Qdrant are tried first, then local fallback silently.

## 3. Technology stack (verified installed in the active interpreter, Python 3.12.10)

FastAPI 0.115.0, Pydantic 2.9.2 / pydantic-settings 2.5.2, SQLAlchemy 2.0.35 + asyncpg + Alembic 1.13.3, qdrant-client 1.19.1, opensearch-py (client only; no server), sentence-transformers 6.1.0, torch 2.14.1 (CPU), transformers 5.18.0, google-generativeai 0.8.4, httpx 0.27.2, structlog, pytest 9.1.1 / pytest-asyncio. Also installed but optional: openai 2.45, langchain-openai (unused by code I saw).

## 4. Repository map

| Path | Role |
|---|---|
| `backend/main.py` | App factory, lifespan (logging + DB pool), CORS, exception handlers |
| `backend/api/v1/{health,retrieval,research,sources,router}.py` | HTTP routes |
| `backend/core/{config,logging,exceptions}.py` | `Settings` (pydantic-settings, root `.env`), structlog |
| `backend/db/` | SQLAlchemy base, `database.py` (`db_manager`), models: `indexed_documents`, `research_queries`, `research_sessions` |
| `backend/alembic/versions/20260929_..._create_initial_tables.py` | single migration |
| `backend/retrieval/` | `passage_chunker`, `bm25_retriever`, `local_bm25`, `dense_retriever`, `hybrid_retriever`, `indexing_service`, `opensearch_schema`, `qdrant_schema`, `models` |
| `backend/embedding/` | `SentenceTransformerProvider`, `MockEmbeddingProvider`, `OpenAIEmbeddingProvider`, `factory.py` |
| `backend/ranking/` | `CrossEncoderReranker`, `RankingService`, `EvidenceSelector`, `evidence.py` |
| `backend/query/` | `intent`, `complexity`, `decomposer`, `merger`, `planner`, `models` |
| `backend/synthesis/` | `answer_service`, `prompt_builder`, `citation_mapper`, `citation_verifier`, `models` |
| `backend/llm/` | `gemini`, `mock`, `ollama`, `openai`, `anthropic` providers + `factory.py` (paid guard) |
| `backend/connectors/` | `arxiv_connector`, `semantic_scholar_connector`, `web_fetcher`, `http_client`, `deduplication`, `ingestion_service` |
| `backend/evaluation/` | `metrics`, `answer_metrics`, `dataset`, `evaluator`, `decomposition_evaluator`, `system_evaluator` (42 KB) |
| `backend/tests/` | 40+ test modules + `integration/` |
| `scripts/` | `evaluate_system.py`, `evaluate_retrieval.py`, `evaluate_decomposition.py`, `phase85_smoke_test.py`, `gemini_validate.py`, `phase8_validate.py`, seed/verify scripts |
| `data/` | `sample_papers.json` (5 papers), `evaluation/{retrieval,answer}_benchmark.json`, **live indexes** `opensearch_local/`, `qdrant_local/` |
| `backend/data/` | **A second, divergent set of indexes** (8 chunks) created when code ran with cwd=`backend/` — see §14 |
| `reports/evaluation/` | Phase 7 reports (2026-09-29) and Phase 8.5 reports (2026-10-01) |
| `docs/` | ARCHITECTURE, EVALUATION, HUMAN_EVALUATION_PROTOCOL, LLM_SYNTHESIS_AND_CITATIONS, QUERY_INTELLIGENCE, RERANKING_AND_EVIDENCE, SOURCE_CONNECTORS, LOCAL_DEVELOPMENT, PHASE_8_5_MANUAL_SETUP, adr/001 (docs not audited line-by-line; README is known-stale) |
| `frontend/` | **Empty** (`.gitkeep` only) |
| `docker-compose.yml`, `infrastructure/`, `services/`, `evaluation/` | Present from Phase 0; not required and not used. **Removed in the pre-publication repository cleanup** (empty or unreferenced scaffold) |

## 5. Main execution paths

**Retrieval** (`services/research_retrieval_service.py::ResearchRetrievalService.retrieve_evidence`): `RetrievalQuery(top_k=candidate_k, alpha=0.5)` → `HybridRetriever.search` fetches `max(3*top_k, 30)` from each retriever concurrently → `reciprocal_rank_fusion` (weights `2(1-α)` / `2α`, k=60; `relative_score_fusion` exists but is never selected by the service) → `RankingService.rerank_candidates` → `EvidenceSelector.select`.
Fallbacks: BM25Retriever and DenseRetriever each try the remote server then local file/embedded storage; on any dense error it returns an **empty list with `metadata.error`**, which hybrid then fuses as "BM25 only" with no signal to the caller.
Note: the planner/hybrid result metadata key is `fusion_method`, while the service reads `fusion_mode` (always defaults to `"rrf"`; cosmetic).

**Answer**: as in §2. `provider_override` in the request can switch LLM provider per call (still passes the paid-provider guard).
Insufficient evidence: if the evidence pack is empty, a canned answer with `insufficient_evidence=true` is returned **without calling the LLM** (VERIFIED by code and by smoke test).
Malformed LLM JSON → answer marked `insufficient_evidence=true` with the parse error in `limitations` (the raw error text is included).

**Ingestion**: `POST /sources/ingest` → `ConnectorIngestionService` → connector → `ResearchDocument` → deduplication helpers → `IndexingService.index_document` (chunk → embed → OpenSearch bulk **and always** local BM25 JSON → Qdrant remote **and always** local Qdrant → optional PostgreSQL `IndexedDocument` row).
Gap: `index_document` mints `uuid4()` per call, so re-ingesting the same paper creates duplicate chunks; deduplication in `connectors/deduplication.py` is not enforced against the index.

## 6. Implemented features

Chunking; BM25 (own Okapi implementation with title/abstract boosting); dense retrieval; RRF + RSF fusion; neural cross-encoder with fallback modes (`hybrid_score` | `error`); evidence selection with diversity and near-duplicate suppression; intent/complexity classification; deterministic + LLM query decomposition; evidence merging with coverage metrics; structured JSON synthesis; citation mapping; deterministic structural verification; Gemini/Mock/Ollama/OpenAI/Anthropic providers with a paid-provider guard; arXiv/Semantic Scholar/web connectors; SSRF checks; evaluation harness for 6 variants with Recall/Precision/MRR/NDCG plus answer metrics.

## 7. Phase-by-phase status

| Phase | Status | Evidence / caveat |
|---|---|---|
| 0 Architecture/scaffolding | **VERIFIED** | Layout, ADR, docs exist. README roadmap numbering (Phase 7 = Frontend, 8 = Evaluation) contradicts actual phases. |
| 1 Backend foundation | **VERIFIED** (with caveat) | App, config, health/ready, SQLAlchemy models, one Alembic migration. Tests pass with infra mocked; live PG on :5433 port is open but I did not run migrations/queries. `/ready` hard-requires OpenSearch+Redis (§14). |
| 2 Retrieval core | **VERIFIED (8.6)** | BM25, dense, hybrid verified against a rebuilt real-embedding index (`verify_retrieval_paths.py`). Previously inconsistent (mock vectors). |
| 2.5 Local infra | **VERIFIED (8.6)** | Local BM25 JSON + embedded Qdrant under a single cwd-independent data root with generation-based manifests. |
| 3 Connectors | **PARTIALLY VERIFIED** | Unit tests with mocked HTTP only. Live arXiv/S2 calls not re-run. Corpus contains 6 arXiv docs, so arXiv worked at least once. S2 and web: no evidence of live success. |
| 4 Rerank + evidence | **VERIFIED** (neural now) | Neural scoring proven 2026-10-01 (`reports/evaluation/phase85_smoke_test.json`). But no benchmark was ever run with it (§12). |
| 5 Synthesis + citations | **PARTIALLY VERIFIED** | Pipeline works and is tested; verification is ID-existence only, not entailment. |
| 6 Query intelligence | **PARTIALLY VERIFIED** | Tested; benefit not demonstrated (Variant E raises recall@5 but its coverage metric is circular, §12). |
| 7 Evaluation | **PARTIALLY VERIFIED** | Harness solid and now reports model/index/dataset identity. A–D re-run with real models (8.6) but on a 7-chunk corpus; E/F not re-run. |
| 8 Real-model support | **VERIFIED** (code); real-model *evaluation* **MISSING** | Config, factory, guard, labeling exist. |
| 8.5 Smoke tests | **PARTIALLY VERIFIED** | Reranker PASS; Gemini ping PASS; the pipeline-level "grounded answer" test **BROKEN** by a script bug (§11); a separate script passed grounded synthesis using *inline* evidence, not the retrieval pipeline. |

## 8. API endpoint inventory (all under `/api/v1`, none authenticated)

| Method | Path | Handler |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | PG + OpenSearch + Qdrant + Redis checks → 200/503 |
| POST | `/retrieval/search` | single-retriever / hybrid search |
| POST | `/retrieval/index` | index a document |
| POST | `/retrieval/chunk` | chunk preview |
| POST | `/research/plan` | analysis + decomposition only |
| POST | `/research/retrieve` | hybrid → rerank → EvidencePack |
| POST | `/research/answer` | full grounded answer |
| GET | `/sources/arxiv/search` | arXiv |
| GET | `/sources/semantic-scholar/search` | Semantic Scholar |
| GET | `/sources/web/fetch` | SSRF-checked fetch |
| POST | `/sources/ingest` | fetch + index |
| — | `/docs`, `/redoc`, `/openapi.json` | OpenAPI |

Error handling: `AppException` → `{"error","code"}`; unhandled → generic 500. However research/source routes wrap everything in `HTTPException(500, detail=f"...{str(e)}")`, which returns raw exception text (may expose paths, provider messages) to clients.

## 9. Database and indexing architecture

_[Rewritten in Phase 8.6]_

* **PostgreSQL**: `indexed_documents`, `research_queries`, `research_sessions`; async via asyncpg, Alembic via psycopg2. `.env` points to localhost:**5433**. `IndexingService` upserts `indexed_documents` via `session.merge()` keyed by the deterministic document id. Not queried in this phase (UNVERIFIED).
* **Data root**: `core/paths.py::resolve_data_root` → `<repo>/data` or `DATA_DIR` (relative → repo root, never cwd).
* **Layout**: `indexes/ACTIVE` (atomic pointer) → `indexes/gen-*/{manifest.json, bm25_index.json, qdrant/}`; `corpus/corpus.json`; `backups/*`.
* **Manifest**: embedding identity (provider, model, dimension, revision, normalization, is_mock), chunking config + chunker version, counts, status, timestamps, corpus sha256. Every vector payload carries `index_generation` and `embedding_model`.
* **BM25**: `LocalBM25Store` JSON per generation (lexical only; embeddings are no longer duplicated into it), tagged with the generation.
* **Dense**: embedded Qdrant per generation, collection `document_chunks`, cosine. A remote Qdrant/OpenSearch server is still tried first if reachable (not used locally); its data is not governed by the manifest.
* **Validation**: `IndexStore.validate_for_read/for_write`, `verify_integrity` (BM25 ids == Qdrant ids == manifest counts; vector validity; tags).
* **Admin**: `scripts/index_admin.py`; library in `retrieval/index_admin.py`.

## 10. Configuration (values excluded; presence checked for `.env`)

| Variable | State in `.env` |
|---|---|
| `LLM_PROVIDER` | `gemini` |
| `GEMINI_API_KEY` | **set** (value not read/printed) |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` (code default matches; `.env.example` still says `gemini-1.5-flash`) |
| `LLM_ALLOW_PAID_PROVIDER` | `false` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | empty |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `local` / `all-MiniLM-L6-v2` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` (device/batch/fallback use code defaults; `RERANKER_FALLBACK_MODE` default `hybrid_score`) |
| `POSTGRES_*` | host localhost, port 5433, user `copilot_user`, db `ai_research_copilot`, password set |
| `OPENSEARCH_*`, `QDRANT_HOST/PORT`, `REDIS_*` | defaults; no server listening on 9200/6333/6379 |
| `SECRET_KEY` | set (not verified to be non-default; production guard exists in `Settings`) |
| `SEMANTIC_SCHOLAR_API_KEY`, `GITHUB_TOKEN` | empty |

Notes: `CELERY_*` in `.env` are unused. `.env` is covered by `.gitignore` (but there is no git repo). `Settings.llm_provider` default is `gemini`, so a fresh checkout without a key fails at generation time, not at startup.

## 11. Tests — commands and current results

_[Updated 8.6: **267 passed, 4 skipped, 36 warnings** (`python -m pytest -q -p no:cacheprovider` from `backend/`, ~3.5 min). The 2026-10-10 audit figure of 239 passed is historical. Opt-in real-model tests: `RUN_REAL_MODEL_TESTS=1 python -m pytest tests/test_real_embeddings.py` → 4 passed. Tests now run in a per-test temp `DATA_DIR`, with an explicit `mock_index` fixture; the suite no longer reads or writes the repository index.]_

Command (run from `backend/`):

```bash
cd backend
python -m pytest -q -p no:cacheprovider
```

**Result on 2026-10-10: `239 passed, 30 warnings in 253.49s`** — 0 failed, 0 skipped. The suite is "offline" in the sense that it needs no API keys, but it is **not hermetic**: it loaded real sentence-transformers weights from the local HF cache and took ~4 minutes. Warnings: qdrant-client server-version check, deprecated `get_sentence_embedding_dimension`, unclosed aiohttp sessions (from the Google SDK path).
The root README says `python -m pytest backend/tests`; I ran from `backend/` because `pythonpath=["."]`/`testpaths=["tests"]` are set there (not tried from repo root — UNVERIFIED).

Findings from execution artifacts:
* `scripts/phase85_smoke_test.py:522` reads `verif.citation_validity_rate`, but `CitationVerificationResult` has no such field → the "grounded answer" experiment fails with `AttributeError` before evidence was assessed (`reports/evaluation/phase85_smoke_test.json`). This is a **script bug**, not a pipeline bug, and the reported "Evidence Passages: 0" is an artifact of it.
* `scripts/gemini_validate.py` passed (2 real calls, 100% grounding, 2 claims) but used inline evidence and bypasses retrieval/rerank/verifier of the real pipeline.

## 12. Evaluation — methodology and actual results

> **_[Superseded 8.6]_** The table below is the historical 2026-09-29 mock-based run, retained only for the audit trail. Valid current results are in README "Evaluation" and `reports/evaluation/latest.{md,json}`. Archived copies: `reports/evaluation/historical_mock_phase7/` (with NOTICE).

**Dataset** (`data/evaluation/retrieval_benchmark.json`): 25 queries with document-level relevance by arXiv ID (18 queries × 2 aliases, 6 × 4, 1 × 3 — aliases like `2005.11401`/`arxiv:2005.11401`, so effectively 1–2 relevant docs/query). `relevant_passages` are empty everywhere → document-level only. `required_dimensions` are free-text terms. Labels appear to be author-assigned, with no inter-annotator check, no human evaluation (the protocol doc exists but no results).
**Corpus**: **8 documents** (5 benchmark papers, a "FastAPI Live Indexing Verification" test doc, `test.integration.001`, and a fabricated arXiv ID `2608.09650`). Only 7 distinct relevant IDs exist in the benchmark. With 8 documents, BM25 MRR = 1.0 on all 25 queries is close to unavoidable and does **not** indicate retrieval quality.

Latest reports (`reports/evaluation/latest.{md,json}`, 2026-09-29 09:27):

| Variant | R@5 | R@10 | MRR | NDCG@10 | Real model? |
|---|---|---|---|---|---|
| A BM25 | 0.96 | 0.98 | 1.00 | 0.958 | n/a (lexical, valid) |
| B Dense | 0.26 | 0.40 | 0.165 | 0.213 | **No — mock hash embeddings** |
| C Hybrid | 0.74 | 0.86 | 0.641 | 0.640 | **No** (mock dense half) |
| D +Rerank | 0.74 | 0.86 | 0.641 | 0.640 | **No — fallback; identical to C by construction** |
| E Decomp+Hybrid | 0.94 | 0.94 | 0.673 | 0.723 | No |
| F Full | 0.86 | 0.86 | 0.657 | 0.692 | No — MockLLM |

Interpretation: the "BM25 beats hybrid" ablation is an artifact of the mock embedder. The report itself states "Reranker Status: fallback_active (Neural: False)" and "LLM Status: mock", but the README table presents these as headline results with no such banner on the rows. Answer metrics (claim support 1.0, citation validity 1.0, etc.) come from MockLLM output and a structural verifier, and are tautological.
`Evidence Dimension Coverage` is substring/keyword presence in retrieved text, so it rewards retrieving more text rather than answering correctly. No confidence intervals; n=25.
Reproducibility: report JSONs include timestamps and environment flags, but no corpus hash, embedding-model identity, or random seeds. `reports/evaluation/` contains three near-identical runs on the same minute window.

**No benchmark has been run with the real embedder, the real reranker, or the real LLM.**

## 13. Real-model validation status (current)

| Capability | Status | Evidence |
|---|---|---|
| `torch`/`sentence-transformers` installed | **VERIFIED** | `pip list` in the active interpreter (torch 2.14.1, ST 6.1.0). Earlier "missing" claim is stale. |
| Neural cross-encoder inference | **VERIFIED (2026-10-01)** | Smoke report: mode `neural`, cold 14.5 s, warm p50 ≈ 284 ms; RAG passage scored +8.7 vs −11.4 for distractors; 5/5 hit@1 on a 5-query mini check. I did not re-run it. |
| MiniLM embedder loads offline from HF cache | **VERIFIED (today)** | Loaded with `HF_HUB_OFFLINE=1`. |
| Real dense retrieval against current index | **VERIFIED (8.6)** | Index rebuilt with real vectors; dense scores match independently recomputed cosine (max err 6.4e-8). Previously broken (cos 0.03). |
| Gemini API connectivity + generation | **VERIFIED (2026-10-01)** | Two independent reports; model `gemini-3.5-flash-lite`. Not re-run today (no live calls made). |
| Gemini through the full pipeline (retrieve→rerank→synthesize→verify) | **NOT VALIDATED** | Smoke test crashed on the script bug; `gemini_validate.py` used inline evidence. |
| Insufficient-evidence behavior | **VERIFIED** | Passed in smoke test and by code reading (short-circuits before LLM). |
| Model-ID safety | **RISK** | `_FREE_TIER_GEMINI_MODELS` in `llm/gemini_provider.py` lists only 1.5/2.0 IDs and is *never referenced*; model validity is only discovered at call time. Docstring and `.env.example` still advertise 1.5. The `google-generativeai` SDK is the legacy Google SDK (Google's newer package is `google-genai`) — UNVERIFIED whether it is still maintained; plan a migration check. |

## 14. Known bugs and limitations

Ordered by severity.

1. _[FIXED in 8.6]_ **Stale mock vectors in the live index (critical, silent).** See §0/§9/§13. Every dense, hybrid, rerank-candidate and full-pipeline result is degraded to roughly BM25-only until the index is rebuilt with the real embedder. No guard detects it.
2. _[FIXED in 8.6]_ **No embedding-model identity or dimension validation** on collections; `DenseRetriever` converts all errors into empty results; the factory falls back to the mock embedder with only an INFO log. This is how #1 happened and will recur.
3. **Citation verification is not semantic** (`synthesis/citation_verifier.py`): `SUPPORTED` ⇔ IDs resolve. `CitationMapper`/prompt rules rely on the LLM's honesty. Needs an entailment/overlap check (e.g., NLI cross-encoder or lexical-overlap threshold) before any grounding claim.
4. _[Mitigated 8.6: README/reports replaced and old runs archived; corpus still tiny]_ **Published evaluation results are non-representative** (mock embedder, fallback reranker, MockLLM, 8-doc corpus). README presents them as "Latest Measured".
5. _[FIXED in 8.6]_ **Two divergent index stores** due to cwd-relative paths (`data/...` hard-coded in `bm25_retriever.py:35`, `indexing_service.py:232,288`, `dense_retriever.py:39`): root `data/` (56 chunks) vs `backend/data/` (8 chunks). Running uvicorn from `backend/` (README Quick Start does exactly this) reads the 8-chunk store, not the 56-chunk one. Also `QdrantClient(path=…)` is created per call; embedded Qdrant is single-process (a `.lock` file exists), so concurrent requests/processes can conflict.
6. **Per-request service construction** in `api/v1/research.py`: `ResearchAnswerService()` → `RankingService()` → new `CrossEncoderReranker()` every request ⇒ lazy model load (~14 s cold) on **every** API call unless the OS caches. `get_embedding_provider()` is cached; the reranker is not. (Static analysis; not measured through HTTP.)
7. **`/ready` semantics**: requires OpenSearch and Redis, neither of which the app uses. In the supported no-Docker setup it always returns 503. Redis is imported in the readiness check only; `redis`, `celery` config are vestigial.
8. **SSRF gaps** (`connectors/web_fetcher.py`): validation resolves DNS once and then `httpx` re-resolves (DNS-rebinding TOCTOU); `follow_redirects=True` with **no re-validation of redirect targets** (a public URL can 302 to `http://169.254.169.254/` or an internal host); IPv4-mapped IPv6 and decimal/octal hosts rely on `ipaddress` only; response size is checked **after** the full body is buffered (`response.content`), so the 5 MB cap does not protect memory; no per-redirect timeout budget. Positives: scheme allow-list, private/loopback/link-local/reserved/multicast block, content-type allow-list, timeout.
9. **Information leakage**: `HTTPException(detail=str(e))` on `/research/*`, `/sources/*`; LLM JSON-parse error text placed in the user-visible `limitations`.
10. **Prompt-injection surface**: passages (including arbitrary web content) are interpolated unfenced into the user prompt; no instruction/data separation or output-claim re-checking beyond ID existence. Evidence trust boundary is undefined for `web` sources.
11. **No authentication/rate limiting** on any endpoint including `/sources/ingest` (can trigger outbound fetches and writes) and `/research/answer` (burns the free Gemini quota). CORS allows credentials with configured origins only (OK).
12. _[FIXED in 8.6]_ **Re-ingestion creates duplicates** (`uuid4` per document); `IndexedDocument` write errors are swallowed with a warning.
13. **arXiv parsing uses stdlib `xml.etree`** although `defusedxml` is a listed dependency (host is trusted; low risk).
14. **Dependency reproducibility**: `requirements.txt` has only lower bounds; no lockfile; `pyproject.toml` (Poetry) and `requirements*.txt` coexist and list different things (`pyproject` has redis/celery lines). `aiosqlite` is used by tests but I did not confirm it is declared. `torch` CPU build was installed manually.
15. **Gemini plumbing**: JSON mode relies on `response_mime_type`; invalid JSON is returned raw and later converted into an "insufficient evidence" answer, which can mask provider errors as content. No retry/backoff or free-tier quota handling in the provider (the HTTP client's backoff is only for connectors). `timeout_seconds` is stored but not applied to the SDK call.
16. **No frontend**, no end-to-end HTTP test with real models, no CI config, no git history, README roadmap/phase labels inconsistent with actual phases, README still lists OpenSearch/Redis/Docker as the stack.
17. `Variant D == Variant C` in the report is a symptom of fallback mode; the fallback reranker returns the incoming order, and `rerank_latency_ms≈2.4` shows no model ran.

## 15. Manual setup still required

1. ~~Rebuild the vector index~~ — **done in 8.6** (`python scripts/index_admin.py rebuild`). Re-run after changing `EMBEDDING_MODEL` or chunk settings; mismatches fail closed.
2. ~~Single data directory~~ — **done in 8.6.** The legacy `data/opensearch_local`, `data/qdrant_local` and `backend/data/` are now unused; archive/remove them manually once you are satisfied with `data/backups/`.
3. Apply `alembic upgrade head` against PostgreSQL :5433 if not already applied (UNVERIFIED — I did not query the DB).
4. Confirm Gemini free-tier quota/limits for `gemini-3.5-flash-lite` (UNVERIFIED beyond the two 2026-10-01 calls).
5. Update `.env.example` (`GEMINI_MODEL`, `POSTGRES_PORT`), and optionally add a `HF_HOME`/offline instruction.
6. Larger, more realistic corpus: ingest ≥ 200–500 abstracts via arXiv/Semantic Scholar for a meaningful benchmark (free; rate-limit friendly).

## 16. Recommended roadmap (smallest sensible set)

**Phase 9 — Index integrity & real-model re-baseline** — _index-integrity items DONE in Phase 8.6; remaining below_
* Record `embedding_model`, `dimension`, `index_version` in a manifest beside the stores; refuse to query/upsert on mismatch; make the mock-embedder fallback an explicit opt-in (`EMBEDDING_PROVIDER=mock`) and surface `embedding_mode` in every response.
* Single `DATA_DIR`; singleton reranker/retrievers via FastAPI lifespan/dependency injection; warm-up at startup.
* Rebuild the index; re-run the benchmark with real embedder + real reranker; report `is_neural`, model IDs, corpus hash.
* Fix `scripts/phase85_smoke_test.py` field bug and rerun the true end-to-end Gemini test (≤ a few calls).

**Phase 10 — Evaluation credibility**
* Corpus of several hundred docs; passage-level labels for a held-out subset; 50+ queries; bootstrap CIs; at least one LLM-or-human grounded-answer review (use `HUMAN_EVALUATION_PROTOCOL.md`).
* Add a semantic claim–evidence check (NLI cross-encoder or span-overlap) to `CitationVerifier`; report "structural validity" and "entailment support" separately.
* Compare BM25 / dense / hybrid / +rerank honestly; keep MockLLM only for CI.

**Phase 11 — Product: minimal research UI + hardening**
* One React page: query box → plan → evidence → answer with clickable citations and a process/latency panel; consume `/research/answer`.
* Hardening: redirect-revalidating SSRF fetch with streamed size cap, error-detail sanitization, simple API-key/rate limit, `/ready` that checks only the stores actually configured, pinned lockfile, `README` rewrite with real screenshots and the *new* measured table.

Skip agents, fine-tuning, Kubernetes, paid services.

## 17. Files to read before changing key subsystems

* Embeddings/index: `backend/embedding/factory.py`, `retrieval/indexing_service.py`, `retrieval/dense_retriever.py`, `retrieval/qdrant_schema.py`, `retrieval/local_bm25.py`, `retrieval/bm25_retriever.py`.
* Retrieval pipeline: `services/research_retrieval_service.py`, `retrieval/hybrid_retriever.py`, `ranking/*`.
* Answering: `synthesis/answer_service.py`, `prompt_builder.py`, `citation_mapper.py`, `citation_verifier.py`, `synthesis/models.py`.
* Providers: `llm/factory.py`, `llm/gemini_provider.py`, `llm/mock_provider.py`.
* Security: `connectors/web_fetcher.py`, `connectors/http_client.py`, `api/v1/sources.py`.
* Evaluation: `evaluation/system_evaluator.py`, `evaluation/metrics.py`, `data/evaluation/*.json`, `scripts/evaluate_system.py`.
* Config: `core/config.py`, `.env.example`.
* Tests that pin behavior: `tests/test_hybrid_fusion.py`, `test_reranker.py`, `test_citation_verifier.py`, `test_phase8_gemini.py`, `test_research_answer_api.py`, `test_evaluation_runner.py`.

## 18. Commands for running and debugging locally

_[8.6] Index/diagnostic commands: `python scripts/index_admin.py status|verify|rebuild`, `python scripts/verify_retrieval_paths.py`, `GET /api/v1/retrieval/index/status`. These work from any directory; data resolves from `DATA_DIR` or `<repo>/data`._

```powershell
# Tests (≈4 min; loads local HF weights)
cd backend            # from the repository root
python -m pytest -q -p no:cacheprovider

# API (run from backend/ — but note data/ paths resolve against cwd; see §14.5)
cd backend            # from the repository root
uvicorn main:app --reload --port 8000

# Seed corpus / verify retrieval (from repo root)
python scripts/seed_all.py
python scripts/verify_retrieval.py

# Evaluation (will report mock/fallback status honestly in its header)
python scripts/evaluate_system.py
python scripts/evaluate_system.py --retrieval-only

# Real-model smoke tests (makes real Gemini calls; consumes free quota)
python scripts/phase85_smoke_test.py
python scripts/gemini_validate.py

# Is the index using real or mock vectors? (cosine near 1.0 = real; ~0 = mock)
#   compare chunk["embedding"] in data/opensearch_local/bm25_index.json
#   with SentenceTransformer("all-MiniLM-L6-v2").encode(chunk["content"], normalize_embeddings=True)
```

Debug tips: set `APP_LOG_LEVEL=DEBUG`; structlog events to grep: `qdrant_remote_unavailable_using_local_storage`, `cross_encoder_model_load_failed`, `reranking_completed` (`fallback=`), `dense_search_error`, `gemini_generation_failed`, `llm_json_parse_failed`.

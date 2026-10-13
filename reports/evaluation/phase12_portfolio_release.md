# Phase 12 — Portfolio Release & Demonstration Readiness

**Date:** 2026-10-10 · **Evidence source:** `reports/evaluation/phase11_final_validation.md` (+ `.json`) · **Phase 11 status preserved:** PASS WITH LIMITATIONS

No new benchmarks and no new Gemini requests were run in this phase.

---

## 1. Files created or changed

| File | Change | Reason |
|---|---|---|
| `README.md` | Rewritten | Portfolio-grade overview: problem, workflow, Mermaid architecture, component roles, connectors, Windows setup, env vars, index management, tests/evaluation reproduction, results, limitations, troubleshooting. Removed obsolete Phase-0 plan, stale commands and the `file:///d:/…` links. |
| `docs/DEMO.md` | Created | Reproducible demo script incl. what is/was not verified live. |
| `docs/PORTFOLIO.md` | Created | Project summary, feature list, resume bullets, interview answer, defensible / not-to-make claims. |
| `docs/ARCHITECTURE.md` | Status note added | Original design document describes a target stack (OpenSearch/Qdrant server/Redis/Celery); note points to the implemented architecture. |
| `docs/LOCAL_DEVELOPMENT.md` | Status note added | Predates index generations; marks OpenSearch/Redis/legacy paths as unused. |
| `docs/LLM_SYNTHESIS_AND_CITATIONS.md`, `RERANKING_AND_EVIDENCE.md`, `SOURCE_CONNECTORS.md` | Links rewritten | `file:///d:/Proj/…` absolute local paths → relative links (leaked local path, broke on other machines). |
| `.gitignore` | Additive rules | `data/indexes/`, `data/backups/`, legacy stores, `backend/data/`, `*.lock`, `frontend/dist/`. |
| `backend/retrieval/dense_retriever.py` | One-line fix | See §6. |
| `backend/tests/test_phase10_hardening.py` | +1 regression test | Event loop stays responsive during a slow first model load. |
| `reports/evaluation/phase12_portfolio_release.md` | Created | This report. |

Not changed: ranking semantics, evidence selection, corpus, active index, benchmark labels, `.env`.

## 2. Setup and demonstration instructions verified

| Instruction | How verified |
|---|---|
| `alembic upgrade head` target is current | `alembic current` → `64466da61299 (head)` (read-only, run this phase) |
| `python scripts/index_admin.py verify` / `status` | Run this phase: `ok: true`, 41 docs / 41 chunks, `compatible: true`, `all-MiniLM-L6-v2`, `is_mock: false` |
| `python -m uvicorn main:app --port 8000` from `backend/`; `npm run dev`; `/health`, `/ready`, `/retrieval/index/status` | Executed in Phase 11 (live; `/ready` returned `ready` with OpenSearch/Qdrant/Redis `skipped`) |
| `evaluate_system.py -v a,b,c,d`, `gemini_e2e_validate.py`, `measure_latency.py`, `verify_retrieval_paths.py` | Executed in Phases 8.6–11 |
| Backend/frontend test commands | Executed in Phase 11 (325 / 4 skipped; tsc, build, 10 tests) |
| `python -m venv`, `pip install -r backend/requirements-dev.txt` | **Not executed** (would modify the environment); verified by file inspection only (`requirements-dev.txt` → `-r requirements.txt` + pytest/aiosqlite) |
| Demo steps 1–6 | Executed in Phase 11 through the live UI with the mock LLM (DOM-driven); real Gemini was exercised once via the script, **not** through the browser |
| Demo step 7 insufficient-evidence | UI rendering verified with a fixture only; live outcome untested and labelled so in the guide |

Dev servers: none were started in this phase; ports 8000 and 5173 confirmed free at the end.

## 3. Documentation and architecture updates
* Mermaid diagram in the README, drawn from the actual code path (`research.py` → `ResearchAnswerService` → `ResearchPlanningService` → per-sub-question `ResearchRetrievalService` [BM25 ∥ dense → RRF → cross-encoder → `EvidenceSelector`] → `EvidenceMerger` → Gemini → `CitationMapper`/`CitationVerifier`), with persistence (`data/indexes/<generation>`, PostgreSQL registry) and connectors. Optional LLM decomposition call is shown as a dotted edge.
* Unused services (OpenSearch, Qdrant server, Redis, `docker-compose.yml`) are called out explicitly as not part of the running setup.
* A finding made while documenting: with the default `decomposition_mode=auto`, **complex questions make an extra Gemini call** for decomposition (`query/decomposer.py::decompose`). The Phase 11 real run used deterministic mode (1 call). This is now stated in the README and demo guide because it affects Free Tier quota.

## 4. Results and screenshots

All results are taken from the Phase 11 report and reproduced in the README: backend **325 passed / 0 failed / 4 skipped (opt-in; 4/4 pass when enabled) / 42 warnings / 179.6 s**; frontend `tsc` clean, build OK, **10** tests; **93** focused security tests; retrieval table (25 queries, 41 documents / 41 chunks, unchanged labels) with the paired-bootstrap caveat — reranking is **not** claimed superior (NDCG@10 +0.014, CI [0.000, +0.033], 3 wins / 0 losses / 22 ties); labels originally written for 7 documents; one real Gemini request PASS (20.5 s, 5 passages, 7 claims, 2 citations); latency: reranker 4.6 s warm / 25 pairs, API warm 4.2 s p50 (mock LLM), first request 57.8 s (n=1); manual citation review = **single reviewer**, 7 claims, not an error-rate estimate; verification is **structural**.

**Screenshots: none obtained.** The in-app browser returned image-capture timeouts throughout Phases 9–11, and no available tool could write an image file. No mock-up or placeholder was created. `docs/DEMO.md` contains exact manual capture instructions (desktop ≥1280 px, mobile 375×812, target filenames).

## 5. Security and release hygiene

| Finding | Severity | Action |
|---|---|---|
| `.env` contains a real Gemini key and database password | Expected | Verified `.env` is in `.gitignore`; values never printed. **The directory is not a Git repository**, so ignore rules are not yet in force — if you run `git init`, check `git status` before the first commit. |
| Secret scan (pattern + exact configured key value) over repository, reports and logs | None found | Only hits: obviously fake key strings in `backend/tests/test_phase10_hardening.py` (test fixtures). Key value present in none of 33 report/log files (Phase 11) |
| Absolute local paths (`file:///d:/Proj/…`, user directory) in committed docs/README | Fixed | 3 docs + README rewritten to relative links; remaining machine-specific text is only in `reports/evaluation/evaluation_*` (hashes/ids) and in `data/` store files (not for publication) |
| `docs/LOCAL_DEVELOPMENT.md` shows the template-default DB password string | Low | Labelled as template default in the status note; set your own in `.env`. Not a real credential if you changed it — I did not inspect your value |
| Generated/local artifacts not ignored (`data/indexes`, `data/backups`, legacy stores, `.lock`) | Medium (repo bloat, ~hundreds of KB–MB of sqlite/JSON, would publish local data) | `.gitignore` extended; nothing deleted |
| Stale scaffolding: `docker-compose.yml`, empty `infrastructure/`, `services/`, `evaluation/` dirs | Low (misleading) | Documented as unused; not removed (no deletions without your say-so) |
| `frontend/package-lock.json`, `node_modules/` | – | lockfile tracked; `node_modules/` and `dist/` ignored |
| No authentication / rate limiting | Known limitation | Documented; not implemented (out of scope) |
| `requirements.txt` has only lower bounds, no lock for Python | Low | Documented limitation in handover; reproducibility relies on the versions recorded in Phase 11 |

Nothing was published, pushed or uploaded; no repository was initialized.

## 6. Cold start and the evidence-selector floor

**Investigation (no new experiments; Phase 11 server log of the 57.8 s first request):**

| Time | Event |
|---|---|
| 08:11:18 | Server started; request sent ≈ 08:11:20 |
| 08:12:10 | `loading_sentence_transformer_model` — ≈ 50 s elapsed *before* the model load began |
| 08:12:11 | OpenSearch probe reports `request:46.672s` despite a 2 s timeout |
| 08:12:14–19 | Cross-encoder loads in 0.3 s, first inference 5.6 s |

Cause: the first request triggers `import sentence_transformers`/torch/transformers (≈ 50 s on a cold Windows disk cache) **synchronously inside the async request handler** (`DenseRetriever.search` → `IndexStore.validate_for_read` → `provider.dimension`). That froze the whole event loop, which is also why the 2 s OpenSearch timeout appeared as 46 s. The model weights themselves load in about 1.4 s (embedding) and 0.3 s (cross-encoder); the reranker's first inference adds ≈ 5 s.

**Fix (small, low-risk):** `validate_for_read` now runs via `asyncio.to_thread`, so the event loop (health checks, other requests) is no longer frozen during the first load. Regression test `test_dense_search_model_load_does_not_block_event_loop` (a provider that blocks 0.6 s; asserts the loop heartbeat gap stays < 0.3 s) passes, together with the dense-retriever, integration and index-integrity subsets (8 passed). **It does not make the first answer faster** — total cold latency is unchanged; no start-up warm-up was added, as instructed. The README states this accurately.

**Evidence selector:** `EvidenceSelector.select` applies `min_score` only if configured (`EVIDENCE_MIN_SCORE` is empty by default), so no relevance floor exists: in the Phase 11 real answer two of five passages had reranker scores ≈ −9.4 and still reached the prompt. Documented in README limitations and the Phase 11 report; no threshold was tuned because there is no evaluation protocol for choosing one.

## 7. Unresolved blockers
None blocking a portfolio demonstration. Open items (all documented): no screenshots in the repo; no authentication/rate limiting; structural-only citation verification; 41-document, partially labeled corpus; ≈ 1-minute cold first request; no Git repository (so ignore rules are untested); stale scaffolding files left in place; Python dependencies not pinned.

## 8. Project description and resume bullets

**Description.** AI Research Copilot is a local, citation-first research assistant: hybrid BM25 + dense retrieval, neural cross-encoder reranking and Gemini Free Tier synthesis, with a React interface that lets a reader open the exact passage behind every claim. It is engineered around integrity and honesty — manifest-validated index generations that fail closed, sanitized provider errors, SSRF-hardened ingestion, and evaluation reported with its limits.

**Resume bullets**
* *Architecture:* Built an end-to-end research-QA system (FastAPI, React/TypeScript, local BM25 + embedded Qdrant, MiniLM, cross-encoder reranker, Gemini Free Tier) with generation-based index manifests that validate embedding model identity and fail closed, after finding and repairing an index populated with same-dimension mock vectors.
* *Retrieval engineering:* Implemented hybrid BM25 + dense retrieval with RRF fusion, neural reranking and evidence selection; hardened the service (per-redirect SSRF validation, streaming size limits, sanitized provider-error taxonomy, shared model lifecycle) with 325 offline backend tests and 10 frontend tests.
* *Evaluation:* Built a reproducible Recall@k/MRR/NDCG harness with paired bootstrap comparisons; on a 41-document corpus (labels written for 7), hybrid + rerank reached NDCG@10 0.994 with intervals overlapping simpler baselines, and documented that citation verification is structural rather than semantic.

Full list of defensible and indefensible claims: `docs/PORTFOLIO.md`.

## 9. Commands to reproduce the demonstration

```powershell
# one-time (see README for details)
copy .env.example .env                       # then set POSTGRES_* and GEMINI_API_KEY (or LLM_PROVIDER=mock)
pip install -r backend/requirements-dev.txt
cd backend; alembic upgrade head; cd ..
python scripts/index_admin.py rebuild
python scripts/index_admin.py verify

# terminal 1
cd backend
python -m uvicorn main:app --port 8000

# terminal 2
cd frontend
npm install
npm run dev                                   # open http://localhost:5173

# health / index
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/ready
curl http://localhost:8000/api/v1/retrieval/index/status

# question to ask in the UI
#   How does dense passage retrieval compare with BM25 keyword search?
```
Then follow `docs/DEMO.md` steps 4–7.

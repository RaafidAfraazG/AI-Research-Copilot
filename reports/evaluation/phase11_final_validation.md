# Phase 11 — Final Validation Report

**Date:** 2026-10-10 · **Scope:** validation only (no feature work) · **Final status: PASS WITH LIMITATIONS**

Machine-readable summary: `phase11_final_validation.json`. Raw artifacts: `reports/evaluation/phase11/`.

---

## 0. Verdict and justification

| Question | Answer |
|---|---|
| Does the full pipeline run end to end with real models? | **Yes.** One real Gemini Free Tier request passed all 11 automated checks (real MiniLM embeddings, neural cross-encoder, Gemini synthesis, structural citation verification). |
| Is the code regression-clean? | **Yes.** Backend 325 passed / 0 failed / 4 skipped (opt-in); the 4 skipped pass when enabled. Frontend type-check, build and 10 unit tests pass. |
| Are security protections for the tested threats effective? | **Yes for the tested cases** (mocked network; 93 focused tests). Not a claim of complete security. |
| Does the benchmark prove retrieval quality? | **No.** 25 queries over a 41-document corpus with labels written for 7 documents. It shows the pipeline behaves sensibly; it does not rank methods with confidence. |
| Is citation verification semantic? | **No.** It is a structural evidence-id check. A single-reviewer manual sample of 7 claims found all supported by the cited passage, which is not a general guarantee. |
| Production ready? | **No** (no authentication/rate limiting, 58 s cold first request, tiny corpus, no entailment check). |

Why not FAIL: no test failures, no material defect was found that invalidates results, and the real-provider path was demonstrated. Why not plain PASS: the limitations above are material to any claim of research-search quality or production readiness.

---

## 1. Environment (no secrets)

| Item | Value |
|---|---|
| OS / hardware | Windows 11, 4 logical CPUs, CPU-only (no CUDA) |
| Python / key libs | 3.12.10; torch 2.14.1+cpu; sentence-transformers 6.1.0; qdrant-client 1.19.1; FastAPI 0.115.0 |
| Node / frontend | Node 20.19.4; React 19, Vite 8, TypeScript, Tailwind 3 |
| Embedding model | `all-MiniLM-L6-v2`, 384-d, normalized, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, real (not mock) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, neural, CPU |
| LLM (real run) | Gemini Free Tier, model `gemini-3.5-flash-lite` (as configured; never switched) |
| Active index | `gen-20261010T080354Z-16e621`, complete, 41 documents / 41 chunks, chunking 512/50 `passage_chunker/v1` |
| Prior generations | `gen-…34c4e8` (7 docs), `gen-…fcfe85` (7 docs) retained; legacy stores and backups untouched |

**Pre-validation audit (all confirmed by inspection/tests):** active manifest is real and complete; BM25, dense, hybrid and rerank all resolve the same active generation through `IndexStore`; mock embeddings are reachable only with `EMBEDDING_PROVIDER=mock`; `MockLLMProvider` appears in non-test code only behind `LLM_PROVIDER=mock` and as the explicit, report-labelled default of the offline `SystemEvaluator`; the Gemini script makes exactly one call with no retry; error bodies are `{error, code, request_id}`; `/ready` checks PostgreSQL + index + LLM config (live check returned `ready` with OpenSearch/Qdrant/Redis `skipped`); the API describes verification as `structural_evidence_id_check`, `semantic_entailment_checked=false`.

---

## 2. Backend tests

```powershell
cd backend
python -m pytest -p no:cacheprovider -q
```

| Metric | Value |
|---|---|
| Passed / failed / skipped | **325 / 0 / 4** |
| Warnings | 42 (qdrant-client server-version check, aiohttp `enable_cleanup_closed` deprecation, deprecated sentence-transformers method name, unclosed aiohttp sessions from tests that build real retrievers without closing them) |
| Runtime | 179.6 s |
| Skipped | 4 opt-in `real_model` tests. Run separately with `RUN_REAL_MODEL_TESTS=1 python -m pytest tests/test_real_embeddings.py` → **4 passed** (46 s) |

Prerequisites/notes: the suite needs no API key or internet, but it is **not hermetic** — several tests (e.g. `test_system_evaluator_variant_d_hybrid_reranker`, 39.6 s) load the real cross-encoder weights from the local Hugging Face cache. Tests run in per-test temporary `DATA_DIR`s and never touch the repository index (legacy stores unchanged).

## 3. Frontend tests

```powershell
cd frontend
npx tsc --noEmit        # clean
npm run build           # success, 1.9 s; JS 377 kB (117 kB gzip), CSS 17.5 kB
npx vitest run          # 10 passed
```

---

## 4. Retrieval benchmark

**Dataset:** `retrieval_benchmark.json` v1.0, 25 queries, sha256 `f0804b35…`, labels unchanged, document-level relevance (1–2 relevant documents per query; 7 distinct relevant documents).
**Corpus:** 7 original documents (5 sample papers, 1 real arXiv abstract, 1 test-fixture document referenced by a label) + 34 arXiv distractor papers fetched by `scripts/build_expanded_corpus.py` (fixed id list, via the existing connector, title+abstract). Original corpus kept as `data/corpus/corpus_v1_7docs.json`.
**Pipeline:** real MiniLM vectors, local BM25, RRF fusion (α=0.5), neural cross-encoder over top-25 candidates. LLM not involved.

| Variant | R@1 | R@5 | R@10 | P@5 | MRR | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| A BM25 | 0.860 | 0.940 | 0.960 | 0.232 | 1.000 | 0.954 | 0.962 |
| B Dense (MiniLM) | 0.760 | 0.980 | 0.980 | 0.248 | 0.933 | 0.925 | 0.925 |
| C Hybrid (RRF) | 0.860 | 0.960 | 1.000 | 0.240 | 1.000 | 0.964 | 0.980 |
| D Hybrid + neural rerank | 0.860 | 1.000 | 1.000 | 0.256 | 1.000 | 0.994 | 0.994 |

For comparison, the earlier 7-document run (Phase 8.6, same labels): R@5 = R@10 = 1.0 for all variants (saturated), NDCG@10 0.994 / 0.971 / 0.997 / 1.000.
Recall@1 is capped below 1.0 because many queries have 2+ relevant documents.

**Paired query-level comparison on NDCG@10** (25 queries, 2000-sample bootstrap, seed 7; descriptive only):

| Comparison | Mean Δ | 95% CI | wins / losses / ties |
|---|---|---|---|
| Hybrid − BM25 | +0.017 | [−0.001, +0.046] | 2 / 1 / 22 |
| Hybrid − Dense | +0.055 | [+0.002, +0.114] | 5 / 1 / 19 |
| Hybrid+Rerank − Hybrid | +0.014 | [0.000, +0.033] | 3 / 0 / 22 |
| Dense − BM25 | −0.037 | [−0.106, +0.021] | 2 / 4 / 19 |

**What can and cannot be concluded**
* Supported: hybrid retrieval was not worse than either single retriever and was modestly better than dense-only on this set; reranking never lowered NDCG@10 (0 losses). The interval for hybrid-vs-dense barely excludes zero; the others include or touch zero.
* **Not supported:** that any method is superior in general. With 25 queries, 22 of 25 query-level ties and a 41-document corpus, the evidence is weak.
* **Label incompleteness:** labels were written for 7 documents. Distractors such as BERT/T5 (transformers), PaLM/Switch/Flan (“scaling”) are topically adjacent and unlabeled; some misses (`ret_24_trend_neural_reranking_advances`, `ret_25_trend_reasoning_scaling_laws`) are partly artifacts of that. Scores are therefore conservative, and this was not corrected because labels may not be edited to change scores.
* Still a **small-corpus benchmark** (41 chunks, one chunk per document). It does not establish general research-search quality.
* Variants E (decomposition) and F (full pipeline) were not re-run: they use the mock LLM in the offline evaluator and would not measure a real model.

Latency fields in the evaluation reports include the cold model load in the first query; use section 7 for latency.

---

## 5. Gemini evidence and citation quality

### 5.1 Real vs mocked runs (this phase)

| Run | Provider | Real? | Outcome |
|---|---|---|---|
| `gemini_e2e_validate.py` (1 request) | `gemini` / `gemini-3.5-flash-lite` | **Yes** | **PASS** — 20.5 s, 1 LLM call, plan: dependency-aware, 3 subqueries (COMPARATIVE); 5 evidence passages / 2 distinct sources; neural reranker active; 7 claims, 2 citations; structural verification valid 7/7; tokens 2011 in / 1214 out |
| Frontend/API workflow (live backend) | `mock` / `mock-research-model-v1` | **No** (labelled “MOCK PROVIDER” in UI) | Used only for repeatable integration checks and latency of the non-LLM stages |
| Error/insufficient-evidence UI states | browser `fetch` fixtures | **No** | Rendering verified only |

Only **one** real Gemini request was made in this phase. A Phase 10 run earlier the same day (different corpus) also passed and is not counted here.

### 5.2 Automated structural checks (real answer)
All claims carry ≥1 citation; every citation id resolves to an evidence passage; no unresolved evidence ids; no verification errors or warnings; sections are consistent with the verified top-level claims (Phase 10 fix confirmed on a real answer).

### 5.3 Manual sampled semantic review (7 of 7 claims; one reviewer, not independent)
Each claim was read against the cited passage (`phase11/gemini_answer.json`).

| Claim | Cites | Judgement |
|---|---|---|
| BM25 commonly used to find candidate passages | DPR passage | Supported verbatim in substance |
| DPR: two BERT encoders, 768-d space, dot-product scoring | DPR passage | Supported (numbers match) |
| Hybrid BM25+DPR yields superior recall | DPR passage | Supported (restates passage) |
| LLMs store knowledge but cannot update it, hallucinate, lack citations | RAG passage | Supported |
| RAG pairs seq2seq with dense retriever over external index | RAG passage | Supported |
| RAG more factual/specific/diverse; index swappable w/o retraining | RAG passage | Supported |
| DPR top-20 accuracy 78.4% vs BM25 59.1% on Natural Questions | DPR passage | Supported (numeric claim matches passage exactly) |

Findings:
* No unsupported numerical, comparative or causal claims were found in this sample.
* The **summary paragraph** is not claim-linked and is not covered by verification (its content matches the claims, but the check is manual only).
* 3 of the 5 evidence passages (RRF, “Attention Is All You Need”, listwise reranking paper) were retrieved but never cited; two had strongly negative reranker scores. Not an error, but shows evidence selection has no relevance floor.
* The model produced one non-standard claim id (`claim_001_perf`); handled correctly.
* The corpus passages for the five sample papers are project-authored summaries, not verbatim paper abstracts, so “grounded” means grounded in those texts.

### 5.4 Limits
Verification is structural: `SUPPORTED` = every cited evidence id exists in the retrieved set. No entailment model is used, and N=7 from a single answer says nothing about error rates.

---

## 6. Application workflow and UI validation

Live backend (mock LLM) + Vite dev server, driven through the actual page in the in-app browser (DOM inspection; screenshots were not obtainable because the pane could not render images).

| Check | Desktop 1280 px | Narrow 375 px |
|---|---|---|
| Health + index status in sidebar (“Backend connected”, 41 docs) | ✔ | ✔ (menu sheet) |
| Empty-input validation | ✔ | – |
| Loading state → answer, plan (2 sub-questions, intent COMPARATIVE) | ✔ | ✔ |
| Claims with citation chips and backend status | ✔ | ✔ |
| Citation → evidence panel with passage text, ranking score, safe external link | ✔ docked `complementary` | ✔ modal `dialog`, focus on close, Esc closes |
| “Cited by” → claim highlight/scroll; panel stays open on desktop | ✔ | ✔ closes panel |
| No horizontal overflow | ✔ | ✔ |
| Error fixtures: 429, index missing, leaky 500, malformed 200, network down | ✔ safe messages, query kept, no leaked text | – |
| Insufficient-evidence fixture | ✔ | – |

Limitation: after a *live* viewport change the pane did not deliver media-query change events, so the docked/overlay switch was verified only on fresh loads at each width. Screenshots/visual polish were not validated. Real-Gemini output was not rendered through the browser (the single real request was spent on the script).

---

## 7. Security validation (93 focused tests, all mocked/fixtures)

`tests/test_phase10_hardening.py`, `test_web_fetcher.py`, `test_http_client.py`, `test_index_integrity.py` → **93 passed** (also part of the 325).

| Control | Result |
|---|---|
| Redirect to private/loopback/link-local/metadata, `::ffff:` mapped, alternate numeric IPs | Blocked; the private hop is never requested; SSRF errors are not retried |
| `file://`, `ftp://`, `gopher://`, credentials in URL, redirect to unsupported scheme | Rejected |
| Oversized streamed body / declared Content-Length / redirect loop | Aborted promptly (<50 of 10 000 chunks read) / rejected / bounded |
| Safe public redirect | Works; connects to validated IP with original Host/SNI |
| API errors | Stable `{error, code, request_id}`; 429/502/503/504 provider categories; no keys, paths, tracebacks, DSNs (fixtures containing fake keys/paths) |
| Log redaction | Configured secret values and credential patterns redacted; repo/home paths normalised |
| Client lifecycle / timeouts | Retriever/indexer `aclose()` awaited; service registry closed; connect-timeout cap and total deadline implemented and exercised |
| Real-key leak scan | The configured key value appears in none of 33 report/log files; no `AIza…` strings in them |

**Not guaranteed / not tested:** DNS rebinding is narrowed (connect to validated IP), not proven eliminated; HTTP proxies via environment are ignored in hardened mode; no authentication or rate limiting on any endpoint; no real private-network probing (deliberately); no fuzzing or dependency vulnerability scan; CORS is permissive to configured origins only.

---

## 8. Latency (CPU only, single machine, small samples)

| Measurement | Value | n |
|---|---|---|
| Embedding model init | 1.4 s | 1 |
| Embedding warm query p50 / p95 | 23.7 / 52.6 ms | 20 |
| Cross-encoder init | 0.3 s (weights already in OS cache; the first-ever load took ~14 s in Phase 8.5) | 1 |
| Cross-encoder first inference, 25 pairs | 5.4 s | 1 |
| Cross-encoder warm inference, 25 pairs p50 / p95 | 4.6 / 6.2 s (≈185 ms/pair; abstract-length passages) | 10 |
| API `/research/answer`, **first request (cold)** | **57.8 s** | 1 |
| API warm total p50 / p95 / max (mock LLM, no decomposition, 25 candidates) | 4.2 / 8.8 / 8.8 s | 7 |
| API warm retrieval stage p50 | 4.2 s (reranker dominates) | 7 |
| Real Gemini end-to-end (decomposed) | 20.5 s | 1 |

Interpretation and caveats: reranking 25 candidates dominates warm latency and is high on this CPU; the cold first request (model imports + weight load + first inference + two ≤2 s probes of unused OpenSearch/Qdrant servers) is long, and nothing warms the models at startup. Samples are tiny, measured on one machine while idle, and no throughput or concurrency testing was done (inference is serialized by a lock).

---

## 9. Defects

* **Fixed in this phase:** none required. No test failed, and no defect was found that invalidates the validation results.
* **Tooling added (not defect fixes):** `scripts/build_expanded_corpus.py`, `scripts/measure_latency.py`, `--dump-answer` option on `gemini_e2e_validate.py`.
* **Observed, documented, not fixed (out of scope / below materiality):**
  1. Cold first API request ≈58 s; models are loaded lazily (no startup warm-up).
  2. First request also pays up to ~2 s × 2 timeouts probing unused OpenSearch and Qdrant servers.
  3. Rerank latency of 25 candidates ≈ 4.6 s on this CPU.
  4. No relevance floor on evidence selection: very low-scoring passages can reach the LLM prompt.
  5. Summary text is not covered by citation verification.
  6. Test suite loads real model weights (not hermetic) and emits unclosed-session warnings.

---

## 10. Known limitations and tests not performed
* Citation verification is structural, not semantic entailment; manual review was N=7, single reviewer.
* Corpus: 41 single-chunk abstract documents; labels cover only 7 and are document-level and author-assigned; no human adjudication; no passage-level evaluation.
* Retrieval method ranking is not statistically established; E/F variants not re-evaluated with a real LLM.
* No authentication, no rate limiting, no concurrency/load testing, no long-run reliability test.
* DNS rebinding mitigated, not eliminated.
* No Gemini quota/outage behaviour tested against the real provider (only fixtures); one real request only.
* No screenshots / visual regression; live viewport-resize behaviour not verified; keyboard/screen-reader audit not performed.
* No PostgreSQL migration or session-persistence tests (no session API exists).

## 11. Compliance with constraints
No paid API was used; the only real LLM call was one Gemini Free Tier request. No Docker, Redis, Ollama or new infrastructure was introduced. No index generations, legacy stores, backups or corpus records were deleted (the corpus file was extended; the original is preserved). No benchmark labels were modified. No mock result is presented as a real-provider result.

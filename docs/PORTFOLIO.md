# Portfolio description

Everything below is grounded in the repository and in `reports/evaluation/phase11_final_validation.md`. No adoption, traffic or business-impact claims are made because none exist.

## Summary
**AI Research Copilot** is a local, citation-first research assistant. It answers research questions from an indexed corpus using hybrid lexical + dense retrieval, neural cross-encoder reranking and Gemini synthesis, then exposes every claim’s supporting passage in a React interface. The project emphasizes engineering rigor around the model pipeline — index integrity, honest evaluation, safe failure modes — rather than model novelty.

## Technical feature list
* **Retrieval:** own Okapi-BM25 implementation, sentence-transformers dense retrieval on embedded Qdrant, Reciprocal Rank Fusion and relative-score fusion, neural cross-encoder reranking, evidence deduplication and per-document diversity limits.
* **Query intelligence:** intent/complexity classification, rule-based and LLM decomposition, parallel sub-question retrieval and evidence merging with coverage metrics.
* **Grounded synthesis:** structured-JSON prompting, claim → citation mapping, structural verification, explicit insufficient-evidence handling; provider abstraction with a free-tier guard (Gemini; paid providers blocked by default).
* **Index integrity:** generation-based indexes with manifests recording embedding identity/revision/chunking; fail-closed identity validation; atomic activation; interrupted-rebuild recovery; deterministic ids for duplicate-safe re-indexing.
* **Ingestion:** arXiv, Semantic Scholar and web connectors; deduplication; SSRF-hardened fetching (per-hop validation, IP pinning, streaming size caps).
* **API:** FastAPI, typed schemas, sanitized error taxonomy with request ids, secret-redacting structured logs, liveness vs. readiness semantics matched to the local architecture, shared reranker singleton.
* **Frontend:** React + TypeScript SPA with plan view, claim/citation rendering, evidence inspector with cited-by navigation, responsive layout, accessible dialog behaviour, sanitized Markdown, explicit error/insufficient-evidence states.
* **Evaluation:** Recall@k/MRR/NDCG harness, paired-bootstrap comparisons, reproducible reports carrying model, index and dataset identity; one-request bounded Gemini validation script.

## Resume bullets (choose one or two)
* **Architecture:** Built an end-to-end research-QA system (FastAPI, React/TypeScript, local BM25 + embedded Qdrant, MiniLM, cross-encoder reranker, Gemini Free Tier) with generation-based index manifests that validate embedding model identity and fail closed, after discovering and repairing an index silently populated with mock vectors of the same dimension.
* **Retrieval engineering:** Implemented hybrid BM25 + dense retrieval with RRF fusion, neural cross-encoder reranking and evidence selection; hardened the pipeline (SSRF-safe fetching with per-redirect validation and streaming size limits, sanitized provider-error taxonomy, shared model lifecycle) and covered it with 325 offline backend tests and 10 frontend tests.
* **Evaluation:** Designed a reproducible evaluation harness (Recall@k, MRR, NDCG, paired bootstrap) and reported results candidly: on a 41-document corpus with labels written for 7 documents, hybrid + rerank reached NDCG@10 0.994 but intervals overlapped with simpler baselines; documented the structural-not-semantic limits of citation verification.

## 60-second interview explanation
“A question comes in; the backend analyses its intent and, if it’s complex, splits it into sub-questions. Each one is searched two ways — BM25 for exact terms and MiniLM embeddings in Qdrant for meaning — and the rankings are fused with Reciprocal Rank Fusion. A cross-encoder rescored the top 25 jointly with the query, then I select diverse, non-duplicate passages. Gemini gets only those passages and must return JSON claims that cite evidence ids; I map them to citations and check that every cited id actually exists. The React UI lets you click a citation and read the exact passage, and jump back to every claim that used it. What I’m most careful about is honesty: the status badge is a structural check, not an entailment proof, and my benchmark is small and partly mislabeled by construction, so I report intervals rather than claiming a winner. The most interesting bug was an index full of hash-based mock vectors with the same 384 dimensions as the real model — dimension checks couldn’t catch it, so I added manifests with model identity validation.”

## Defensible claims
* Implements and validates a hybrid-retrieval → neural-rerank → grounded-synthesis → citation-inspection pipeline with real models (MiniLM, MS-MARCO MiniLM cross-encoder, Gemini Free Tier).
* 325 passed / 4 opt-in skipped backend tests; frontend type-check, build and 10 unit tests pass; 93 focused security tests passed (mocked network).
* On the 25-query, 41-document benchmark: BM25 NDCG@10 0.962, dense 0.925, hybrid 0.980, hybrid + rerank 0.994 — with the stated caveats.
* One real Gemini Free Tier end-to-end request passed all automated structural checks (7 claims, 2 citations, 20.5 s).
* Fail-closed index validation, provider-error sanitization and SSRF protections exist and are tested.
* A single-reviewer manual check of 7 claims found them supported by the cited passages.

## Claims **not** to make
* That reranking or hybrid retrieval is *statistically better* (intervals touch zero, 22/25 ties).
* “Verified”/“hallucination-free”/“semantic” citation checking, or any accuracy/precision rate for answers.
* Production readiness, scalability, throughput, or security completeness (no auth/rate limiting; DNS rebinding not eliminated; no load test).
* Performance on real research corpora: the corpus is 41 abstracts and five texts are project-authored summaries.
* Low latency: warm API ≈ 4 s with mock LLM, ≈ 20 s with real Gemini, ≈ 1 minute cold start (n small, CPU-only).
* Support for GitHub connectors, user accounts, saved sessions, streaming, or multi-user use.
* Any users, traffic, adoption or business impact.

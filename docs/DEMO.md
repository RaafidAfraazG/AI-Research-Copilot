# Demonstration script (≈10 minutes)

All steps were exercised during Phases 9–11 on Windows 11 / PowerShell against the 41-document corpus. Steps that were **not** exercised end to end are marked.

**What the system can answer:** only what is in the indexed corpus: 41 paper abstracts/summaries — the RAG, DPR, Sentence-BERT, reciprocal-rank-fusion and Transformer papers, one 2026 listwise-reranking paper, and 34 general ML papers (BERT, GPT-3, T5, ResNet, Adam, LoRA, …). It has **not** searched arXiv or the web for your question. Good questions are about retrieval, RAG, dense vs. sparse search, rank fusion, sentence embeddings.

## 0. Prerequisites
Setup from the [README](../README.md) is complete (`.env`, `alembic upgrade head`, `python scripts/index_admin.py rebuild`).

* **Real model run:** `LLM_PROVIDER=gemini` and a valid `GEMINI_API_KEY` in `.env`. Mind the Free Tier: a complex question may use two Gemini calls (decomposition + synthesis).
* **Offline / deterministic run:** start the backend with `$env:LLM_PROVIDER = "mock"`. The UI footer then says “MOCK PROVIDER — not a real model answer”. Do not present mock output as model output.

## 1. Start the services
```powershell
# Terminal 1
cd backend
python -m uvicorn main:app --port 8000

# Terminal 2
cd frontend
npm run dev                  # http://localhost:5173
```
The first answer after launch is slow (≈1 minute measured: cold torch import and model loading). Do one throw-away question before presenting, or explain it.

## 2. Confirm health and the active index
```powershell
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/ready
curl http://localhost:8000/api/v1/retrieval/index/status
```
Expected: `/health` `ok`; `/ready` `ready` with `postgres`, `index`, `llm` ok and `opensearch`/`qdrant`/`redis` **skipped**; index status shows `all-MiniLM-L6-v2`, 384-d, `is_mock: false`, 41 documents / 41 chunks. In the UI the sidebar shows “Backend connected”, the same counts and the embedding model.

## 3. Ask a question
Open <http://localhost:5173> and use the example **“How does dense passage retrieval compare with BM25 keyword search?”** (or type your own; Ctrl/⌘+Enter submits).
Point out: the loading panel shows only elapsed time — the UI does not claim to know which backend stage is running.

## 4. Inspect the plan
Expand **Research plan**: intent (COMPARATIVE), complexity, strategy, the generated sub-questions with their purposes, “sub-questions with evidence 2 of 2”, distinct sources. This panel only appears when the backend returned a plan.

## 5. Read claims and citations
In **Findings & supporting evidence**: each claim carries numbered citation chips and a status badge. Say explicitly: *“Supported” means the claim cites evidence that exists in the retrieved set — it is a structural check, not proof the passage entails the claim.* The evidence-panel footer repeats this.

## 6. Open the evidence and navigate back
Click a citation number → the evidence panel shows the exact passage, source type, authors/date when available, the ranking score (a reranker score, not a probability), and an “Open source” link (arXiv). Under **Cited by**, click a claim to jump back to it (highlighted). On a narrow window the panel is a modal sheet; Esc closes it.

Compare the claim text to the passage live — e.g. the claim “top-20 accuracy 78.4% vs 59.1% on Natural Questions” appears verbatim in the DPR passage.

## 7. Error and insufficient-evidence states
* **Backend unavailable (verified):** stop the backend (Ctrl+C in terminal 1) and press *Research*. The UI shows “Can’t reach the backend”, keeps your question and offers *Try again*.
* **Index not ready (verified with a fixture, not live):** the UI renders `index_manifest_missing` as “Research index unavailable”.
* **Insufficient evidence (rendering verified with a fixture only):** the retrieval stage always returns its top candidates, so an off-corpus question reaches the LLM, which *may* flag `insufficient_evidence`; this live outcome was **not** tested. Do not promise it in a demo.

## 8. Optional: reproduce the numbers
```powershell
python scripts/evaluate_system.py -v a,b,c,d --output reports/evaluation/repro
python scripts/gemini_e2e_validate.py            # one real Gemini request; exit code 2 if no key
```

## Screenshots
Genuine screenshots could **not** be captured in this environment: the in-app browser tooling returned timeouts for image capture, and no tool available could save images to disk. Nothing in the repository is a mock-up. To capture them yourself:

1. Run steps 1–6 with the real provider.
2. Desktop: browser window ≥ 1280 px wide; capture (a) the initial screen, (b) answer with evidence panel open, (c) the plan expanded.
3. Mobile: DevTools → device toolbar → 375×812; capture the answer and the evidence sheet.
4. Save as `docs/screenshots/<name>.png` and reference them from the README.

## Shutting down
Ctrl+C in both terminals. Nothing else needs stopping (no Docker, Redis or OpenSearch).

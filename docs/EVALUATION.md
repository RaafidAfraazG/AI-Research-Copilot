# AI Research Copilot — Evaluation Methodology & Scientific Benchmark (Phase 7)

## Overview

The primary goal of Phase 7 is to establish a **scientifically structured, reproducible evaluation framework** that quantitatively measures the individual and collective contributions of every major system stage:
* Lexical keyword search (BM25)
* Dense semantic vector retrieval
* Hybrid reciprocal rank fusion (RRF)
* Cross-encoder neural reranking
* Query intelligence and intent-driven decomposition
* Grounded LLM synthesis with claim-level attribution
* Deterministic citation verification

---

## 1. System Variants Evaluated

The benchmark compares 6 progressive architecture variants without altering unrelated components:

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
Retrieval Metrics (Recall, Precision, MRR, NDCG)
        ↓
Evidence Dimension Coverage
        ↓
Answer Synthesis & Claim Extraction
        ↓
Grounding & Citation Verification
        ↓
Latency Profiling & Ablation Report
```

---

## 2. Evaluation Datasets & Ground Truth

Evaluation datasets are located in `data/evaluation/`:
* `data/evaluation/retrieval_benchmark.json`: 25 curated technical queries paired with target document and passage relevance labels.
* `data/evaluation/answer_benchmark.json`: 25 research queries with expected research dimensions, canonical documents, and difficulty ratings.
* `data/evaluation/decomposition_benchmark.json`: 16 multi-intent queries evaluating classification, coverage, and constraint preservation.

### Intent Coverage
Queries systematically span all 8 research intent classes:
1. `FACTUAL` (e.g. attention formulas, parameter definitions)
2. `EXPLANATORY` (e.g. RAG marginalization mechanisms)
3. `COMPARATIVE` (e.g. BM25 vs. DPR, bi-encoder vs. cross-encoder)
4. `CAUSAL` (e.g. dense retrieval out-of-vocabulary failures)
5. `PROCEDURAL` (e.g. multi-stage retrieval implementations)
6. `MULTI_HOP` (e.g. cross-paper architectural interactions)
7. `SURVEY` (e.g. taxonomy of dense retrieval and reranking paradigms)
8. `TREND` (e.g. modular RAG and hallucination mitigation trends)

### Ground Truth Provenance
Ground truth relevance is sourced directly from peer-reviewed scientific literature indexed in the research corpus:
* Lewis et al., 2020 (NeurIPS, arXiv:2005.11401) — RAG architectures
* Karpukhin et al., 2020 (EMNLP, arXiv:2004.04906) — Dense Passage Retrieval
* Vaswani et al., 2017 (NeurIPS, arXiv:1706.03762) — Transformer Attention
* Reimers & Gurevych, 2019 (EMNLP, arXiv:1908.10084) — Sentence-BERT
* Cormack, Clarke & Buettcher, 2009 (SIGIR) — Reciprocal Rank Fusion

---

## 3. Metric Formulations

### 3.1 Information Retrieval Metrics

* **Recall@K**: Proportion of relevant documents retrieved within top $K$ candidates:
  $$\text{Recall}@K = \frac{|\text{Retrieved}_{1..K} \cap \text{Relevant}|}{|\text{Relevant}|}$$
* **Precision@K**: Proportion of top $K$ retrieved items that are relevant:
  $$\text{Precision}@K = \frac{|\text{Retrieved}_{1..K} \cap \text{Relevant}|}{K}$$
* **Mean Reciprocal Rank (MRR)**: Reciprocal rank of the first relevant retrieved document:
  $$\text{MRR} = \frac{1}{\text{rank}_1}$$
* **Normalized Discounted Cumulative Gain (NDCG@K)**: Discounted ranking gain normalized by ideal DCG:
  $$\text{DCG}@K = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
* **Evidence Dimension Coverage**: Fraction of required technical dimensions captured in the retrieved candidate pool or merged evidence.

### 3.2 Groundedness & Citation Metrics

* **Claim Support Rate**: Fraction of extracted factual claims substantiated by evidence:
  $$\text{Claim Support Rate} = \frac{\text{Supported Claims}}{\text{Total Factual Claims}}$$
* **Unsupported Claim Rate**:
  $$\text{Unsupported Claim Rate} = \frac{\text{Unsupported Claims}}{\text{Total Factual Claims}}$$
* **Citation Coverage**: Fraction of claims possessing at least one valid citation pointer.
* **Citation Validity Rate**: Fraction of rendered bibliographic citations that reference authentic retrieved passage identifiers.
* **Citation Precision**: Fraction of citations whose passage text corroborates the asserted claim.
* **Citation Traceability Rate**: Fraction of claims traceable through the full lineage chain:
  $$\text{Claim} \rightarrow \text{Passage ID} \rightarrow \text{Document ID} \rightarrow \text{Canonical URL}$$

---

## 4. Benchmark Execution & Reproducibility

### 4.1 Running the Full System Benchmark

```powershell
python scripts/evaluate_system.py
```

### 4.2 Running Specific Variants or Subsets

* **Retrieval Ablation Only** (Variants A–E):
  ```powershell
  python scripts/evaluate_system.py --retrieval-only
  ```
* **Full Answer & Grounding Pipeline Only** (Variant F):
  ```powershell
  python scripts/evaluate_system.py --answers
  ```
* **Custom Variant Selection**:
  ```powershell
  python scripts/evaluate_system.py --variant hybrid --variant rerank --variant full
  ```
* **Decomposition Intelligence Benchmark**:
  ```powershell
  python scripts/evaluate_decomposition.py
  ```

---

## 5. Environment & Model Transparency

The evaluation framework automatically detects and transparently reports environment capabilities:

1. **Reranker Status**:
   * If `sentence_transformers` is installed and neural weights are loaded: reports `neural_loaded (cross-encoder/ms-marco-MiniLM-L-6-v2)`.
   * If running in fallback mode: reports `fallback_active (hybrid_score)` and logs:
     > *"Cross-encoder benchmark unavailable in current environment. Fallback ranking was used."*
2. **LLM Provider Status**:
   * If using the default offline runner: reports `mock (mock-deterministic-grounded)`.
   * If connected to local Ollama or OpenAI: reports provider name and model identifier.

---

## 6. Output Artifacts

Benchmark runs produce machine-readable and human-readable reports:
* `reports/evaluation/latest.json`: Complete serialized benchmark metrics, distributions, and query-level details.
* `reports/evaluation/latest.md`: Formatted Markdown ablation table and statistical breakdown.
* `reports/evaluation/evaluation_YYYYMMDD_HHMMSS.json`: Timestamped historical archive.

---

## 7. Limitations

* **Sample Size**: The 25-query benchmark is designed for directional verification against the indexed seed corpus; it is not a massive statistical corpus.
* **Cross-Encoder Weights**: If the neural model cannot load and `RERANKER_FALLBACK_MODE=hybrid_score`, a no-op fallback is used and the report says `fallback_active`. A fallback run is never a reranker measurement.
* **Mock LLM**: Default automated runs use deterministic mock synthesis to ensure 100% offline testability without API keys.

---

## 8. Phase 8.6 — Real-Embedding Re-baseline

**What changed.** The persisted index previously held mock hash vectors (same 384
dimensions as MiniLM, so undetectable by dimension checks) and test-suite pollution. The
index was rebuilt from `data/corpus/corpus.json` with the real `all-MiniLM-L6-v2` and is now
governed by a manifest (see ARCHITECTURE §5.5). Reports now include the embedding model and
revision, index generation and counts, reranker model/mode, dataset name/version/sha256,
Recall@1/5/10, Precision@5, MRR and NDCG@5/10.

**Archived, not valid:** `reports/evaluation/historical_mock_phase7/` (mock embedder,
fallback reranker, MockLLM, polluted 56-chunk index). Do not cite.

**Current result** (variants A–D, 25 queries, 7 documents / 7 chunks): every variant has
Recall@5 = Recall@10 = 1.0; MRR 1.000 (BM25), 0.973 (dense), 1.000 (hybrid), 1.000 (hybrid +
neural rerank). The corpus is too small to separate the methods; differences are not
evidence of improvement. A meaningful comparison needs a corpus of hundreds of documents,
passage-level labels and confidence intervals (planned Phase 10).

**Benchmark corpus caveats.** The benchmark labels reference `2608.09650` (a real arXiv
abstract recovered from the legacy index) and `test.integration.001` (a test fixture
document). Labels were deliberately not changed; both documents are in the rebuilt corpus
with their provenance recorded in `corpus.json`.

**Independent path verification:** `scripts/verify_retrieval_paths.py` recomputes cosine
similarity from the stored vectors with a freshly loaded embedder (max |error| 6.4e-8
vs the dense retriever's scores), checks hybrid rank attribution, confirms the neural
cross-encoder is active, and confirms a mock provider is rejected by the real index.

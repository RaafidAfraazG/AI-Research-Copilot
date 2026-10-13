# AI Research Copilot — Evaluation & Ablation Report

**Generated:** `2026-10-10T06:27:43.552932+00:00`  
**Dataset Queries:** `25`  
**Reranker Status:** `neural_loaded (cross-encoder/ms-marco-MiniLM-L-6-v2)` (Neural: `True`)  
**LLM Status:** `mock` (Live Model: `False`)  
**Embedding:** `all-MiniLM-L6-v2` (384d, mock: `False`, rev `1110a243fdf4`)  
**Index generation:** `gen-20261010T062446Z-34c4e8` (7 docs / 7 chunks)  
**Reranker model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` on `cpu`  
**Dataset:** `retrieval_benchmark.json` sha256 `f0804b3532f6` v1.0  

---

## 1. System Retrieval Ablation Study

| Variant | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR | NDCG@5 | NDCG@10 | Dim Coverage | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A: BM25 Lexical Retrieval** | 0.8600 | 1.0000 | 1.0000 | 0.2580 | 1.0000 | 0.9936 | 0.9936 | 0.7863 | 84.91 |
| **Variant B: Dense Vector Retrieval** | 0.8200 | 1.0000 | 1.0000 | 0.2560 | 0.9733 | 0.9708 | 0.9708 | 0.7863 | 201.83 |
| **Variant C: Hybrid Retrieval (BM25 + Dense)** | 0.8600 | 1.0000 | 1.0000 | 0.2560 | 1.0000 | 0.9968 | 0.9968 | 0.7863 | 84.82 |
| **Variant D: Hybrid + Cross-Encoder Reranker** | 0.8600 | 1.0000 | 1.0000 | 0.2560 | 1.0000 | 1.0000 | 1.0000 | 0.7863 | 1386.22 |

> [!NOTE]
> Directional evaluation on a very small corpus; see section 5. Do not generalize.

---

## 2. Statistical Distribution of Retrieval Metrics

| Variant | Metric | Mean | Median | Min | Max | StdDev |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Variant A: BM25 Lexical Retrieval | `recall_at_1` | 0.8600 | 1.0000 | 0.5000 | 1.0000 | 0.2291 |
| Variant A: BM25 Lexical Retrieval | `recall_at_5` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant A: BM25 Lexical Retrieval | `recall_at_10` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant A: BM25 Lexical Retrieval | `mrr` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant A: BM25 Lexical Retrieval | `ndcg_at_5` | 0.9936 | 1.0000 | 0.9197 | 1.0000 | 0.0222 |
| Variant A: BM25 Lexical Retrieval | `ndcg_at_10` | 0.9936 | 1.0000 | 0.9197 | 1.0000 | 0.0222 |
| Variant A: BM25 Lexical Retrieval | `evidence_dimension_coverage` | 0.7863 | 0.8333 | 0.0000 | 1.0000 | 0.2342 |
| Variant B: Dense Vector Retrieval | `recall_at_1` | 0.8200 | 1.0000 | 0.0000 | 1.0000 | 0.2843 |
| Variant B: Dense Vector Retrieval | `recall_at_5` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant B: Dense Vector Retrieval | `recall_at_10` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant B: Dense Vector Retrieval | `mrr` | 0.9733 | 1.0000 | 0.3333 | 1.0000 | 0.1333 |
| Variant B: Dense Vector Retrieval | `ndcg_at_5` | 0.9708 | 1.0000 | 0.5000 | 1.0000 | 0.1036 |
| Variant B: Dense Vector Retrieval | `ndcg_at_10` | 0.9708 | 1.0000 | 0.5000 | 1.0000 | 0.1036 |
| Variant B: Dense Vector Retrieval | `evidence_dimension_coverage` | 0.7863 | 0.8333 | 0.0000 | 1.0000 | 0.2342 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `recall_at_1` | 0.8600 | 1.0000 | 0.5000 | 1.0000 | 0.2291 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `recall_at_5` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `recall_at_10` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `mrr` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `ndcg_at_5` | 0.9968 | 1.0000 | 0.9197 | 1.0000 | 0.0161 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `ndcg_at_10` | 0.9968 | 1.0000 | 0.9197 | 1.0000 | 0.0161 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `evidence_dimension_coverage` | 0.7863 | 0.8333 | 0.0000 | 1.0000 | 0.2342 |
| Variant D: Hybrid + Cross-Encoder Reranker | `recall_at_1` | 0.8600 | 1.0000 | 0.5000 | 1.0000 | 0.2291 |
| Variant D: Hybrid + Cross-Encoder Reranker | `recall_at_5` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant D: Hybrid + Cross-Encoder Reranker | `recall_at_10` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant D: Hybrid + Cross-Encoder Reranker | `mrr` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant D: Hybrid + Cross-Encoder Reranker | `ndcg_at_5` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant D: Hybrid + Cross-Encoder Reranker | `ndcg_at_10` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant D: Hybrid + Cross-Encoder Reranker | `evidence_dimension_coverage` | 0.7863 | 0.8333 | 0.0000 | 1.0000 | 0.2342 |

---

## 4. Latency Profiling

| Variant | Mean Total Latency (ms) | Granular Components |
| :--- | :---: | :--- |
| **Variant A: BM25 Lexical Retrieval** | 84.91 ms | `retrieval_latency_ms`: 84.9ms |
| **Variant B: Dense Vector Retrieval** | 201.83 ms | `retrieval_latency_ms`: 201.8ms |
| **Variant C: Hybrid Retrieval (BM25 + Dense)** | 84.82 ms | `retrieval_latency_ms`: 84.8ms |
| **Variant D: Hybrid + Cross-Encoder Reranker** | 1386.22 ms | `reranking_latency_ms`: 1297.2ms, `retrieval_latency_ms`: 88.9ms |

---

## 5. Methodological Limitations & Integrity Declarations

* Ground truth based on 25 curated technical queries targeting a small indexed corpus.
* Sample size is directional and not statistically conclusive.
* The index contains only 7 documents / 7 chunks; with so few candidates, Recall@5/@10 saturate and cannot separate retrievers. Rank-sensitive metrics (Recall@1, MRR, NDCG) are the informative ones.
* Relevance labels are document-level and author-assigned; no human adjudication or inter-annotator agreement.
* Answer evaluation executed with deterministic Mock LLM provider. Live model variance not measured.

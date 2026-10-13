# AI Research Copilot — Evaluation & Ablation Report

**Generated:** `2026-10-10T08:08:16.681437+00:00`  
**Dataset Queries:** `25`  
**Reranker Status:** `neural_loaded (cross-encoder/ms-marco-MiniLM-L-6-v2)` (Neural: `True`)  
**LLM Status:** `mock` (Live Model: `False`)  
**Embedding:** `all-MiniLM-L6-v2` (384d, mock: `False`, rev `1110a243fdf4`)  
**Index generation:** `gen-20261010T080354Z-16e621` (41 docs / 41 chunks)  
**Reranker model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` on `cpu`  
**Dataset:** `retrieval_benchmark.json` sha256 `f0804b3532f6` v1.0  

---

## 1. System Retrieval Ablation Study

| Variant | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR | NDCG@5 | NDCG@10 | Dim Coverage | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A: BM25 Lexical Retrieval** | 0.8600 | 0.9400 | 0.9600 | 0.2320 | 1.0000 | 0.9536 | 0.9623 | 0.8913 | 99.01 |
| **Variant B: Dense Vector Retrieval** | 0.7600 | 0.9800 | 0.9800 | 0.2480 | 0.9333 | 0.9249 | 0.9249 | 0.8913 | 182.60 |
| **Variant C: Hybrid Retrieval (BM25 + Dense)** | 0.8600 | 0.9600 | 1.0000 | 0.2400 | 1.0000 | 0.9641 | 0.9797 | 0.8913 | 90.83 |
| **Variant D: Hybrid + Cross-Encoder Reranker** | 0.8600 | 1.0000 | 1.0000 | 0.2560 | 1.0000 | 0.9936 | 0.9936 | 0.8913 | 5629.80 |

> [!NOTE]
> Directional evaluation on a very small corpus; see section 5. Do not generalize.

---

## 2. Statistical Distribution of Retrieval Metrics

| Variant | Metric | Mean | Median | Min | Max | StdDev |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Variant A: BM25 Lexical Retrieval | `recall_at_1` | 0.8600 | 1.0000 | 0.5000 | 1.0000 | 0.2291 |
| Variant A: BM25 Lexical Retrieval | `recall_at_5` | 0.9400 | 1.0000 | 0.5000 | 1.0000 | 0.1658 |
| Variant A: BM25 Lexical Retrieval | `recall_at_10` | 0.9600 | 1.0000 | 0.5000 | 1.0000 | 0.1384 |
| Variant A: BM25 Lexical Retrieval | `mrr` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant A: BM25 Lexical Retrieval | `ndcg_at_5` | 0.9536 | 1.0000 | 0.6131 | 1.0000 | 0.1283 |
| Variant A: BM25 Lexical Retrieval | `ndcg_at_10` | 0.9623 | 1.0000 | 0.6131 | 1.0000 | 0.1103 |
| Variant A: BM25 Lexical Retrieval | `evidence_dimension_coverage` | 0.8913 | 0.9000 | 0.5000 | 1.0000 | 0.1325 |
| Variant B: Dense Vector Retrieval | `recall_at_1` | 0.7600 | 1.0000 | 0.0000 | 1.0000 | 0.3571 |
| Variant B: Dense Vector Retrieval | `recall_at_5` | 0.9800 | 1.0000 | 0.5000 | 1.0000 | 0.1000 |
| Variant B: Dense Vector Retrieval | `recall_at_10` | 0.9800 | 1.0000 | 0.5000 | 1.0000 | 0.1000 |
| Variant B: Dense Vector Retrieval | `mrr` | 0.9333 | 1.0000 | 0.3333 | 1.0000 | 0.1863 |
| Variant B: Dense Vector Retrieval | `ndcg_at_5` | 0.9249 | 1.0000 | 0.5000 | 1.0000 | 0.1520 |
| Variant B: Dense Vector Retrieval | `ndcg_at_10` | 0.9249 | 1.0000 | 0.5000 | 1.0000 | 0.1520 |
| Variant B: Dense Vector Retrieval | `evidence_dimension_coverage` | 0.8913 | 0.9000 | 0.5000 | 1.0000 | 0.1325 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `recall_at_1` | 0.8600 | 1.0000 | 0.5000 | 1.0000 | 0.2291 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `recall_at_5` | 0.9600 | 1.0000 | 0.5000 | 1.0000 | 0.1384 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `recall_at_10` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `mrr` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `ndcg_at_5` | 0.9641 | 1.0000 | 0.6131 | 1.0000 | 0.1085 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `ndcg_at_10` | 0.9797 | 1.0000 | 0.7977 | 1.0000 | 0.0574 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `evidence_dimension_coverage` | 0.8913 | 0.9000 | 0.5000 | 1.0000 | 0.1325 |
| Variant D: Hybrid + Cross-Encoder Reranker | `recall_at_1` | 0.8600 | 1.0000 | 0.5000 | 1.0000 | 0.2291 |
| Variant D: Hybrid + Cross-Encoder Reranker | `recall_at_5` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant D: Hybrid + Cross-Encoder Reranker | `recall_at_10` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant D: Hybrid + Cross-Encoder Reranker | `mrr` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant D: Hybrid + Cross-Encoder Reranker | `ndcg_at_5` | 0.9936 | 1.0000 | 0.9197 | 1.0000 | 0.0222 |
| Variant D: Hybrid + Cross-Encoder Reranker | `ndcg_at_10` | 0.9936 | 1.0000 | 0.9197 | 1.0000 | 0.0222 |
| Variant D: Hybrid + Cross-Encoder Reranker | `evidence_dimension_coverage` | 0.8913 | 0.9000 | 0.5000 | 1.0000 | 0.1325 |

---

## 4. Latency Profiling

| Variant | Mean Total Latency (ms) | Granular Components |
| :--- | :---: | :--- |
| **Variant A: BM25 Lexical Retrieval** | 99.01 ms | `retrieval_latency_ms`: 99.0ms |
| **Variant B: Dense Vector Retrieval** | 182.60 ms | `retrieval_latency_ms`: 182.6ms |
| **Variant C: Hybrid Retrieval (BM25 + Dense)** | 90.83 ms | `retrieval_latency_ms`: 90.8ms |
| **Variant D: Hybrid + Cross-Encoder Reranker** | 5629.80 ms | `reranking_latency_ms`: 5473.7ms, `retrieval_latency_ms`: 155.9ms |

---

## 5. Methodological Limitations & Integrity Declarations

* Ground truth based on 25 curated technical queries targeting a small indexed corpus.
* Sample size is directional and not statistically conclusive.
* The index contains only 41 documents / 41 chunks; with so few candidates, Recall@5/@10 saturate and cannot separate retrievers. Rank-sensitive metrics (Recall@1, MRR, NDCG) are the informative ones.
* Relevance labels are document-level and author-assigned; no human adjudication or inter-annotator agreement.
* Answer evaluation executed with deterministic Mock LLM provider. Live model variance not measured.

# AI Research Copilot — Evaluation & Ablation Report (Phase 7)

**Generated:** `2026-09-29T09:27:43.244992+00:00`  
**Dataset Queries:** `25`  
**Reranker Status:** `fallback_active (hybrid_score)` (Neural: `False`)  
**LLM Status:** `mock` (Live Model: `False`)  

---

## 1. System Retrieval Ablation Study

| Variant | Recall@5 | Recall@10 | Precision@5 | MRR | NDCG@10 | Dim Coverage | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A: BM25 Lexical Retrieval** | 0.9600 | 0.9800 | 0.5120 | 1.0000 | 0.9580 | 0.8087 | 165.47 |
| **Variant B: Dense Vector Retrieval** | 0.2600 | 0.4000 | 0.1040 | 0.1651 | 0.2128 | 0.4783 | 127.72 |
| **Variant C: Hybrid Retrieval (BM25 + Dense)** | 0.7400 | 0.8600 | 0.3280 | 0.6414 | 0.6400 | 0.7753 | 114.21 |
| **Variant D: Hybrid + Cross-Encoder Reranker** | 0.7400 | 0.8600 | 0.3280 | 0.6414 | 0.6400 | 0.7753 | 96.80 |
| **Variant E: Query Decomposition + Hybrid** | 0.9400 | 0.9400 | 0.2540 | 0.6727 | 0.7233 | 0.7337 | 232.32 |
| **Variant F: Full Research Pipeline** | 0.8600 | 0.8600 | 0.2800 | 0.6567 | 0.6923 | 0.2827 | 228.13 |

> [!NOTE]
> Directional evaluation across indexed technical corpus. Ground truth relevance explicitly documented.

---

## 2. Statistical Distribution of Retrieval Metrics

| Variant | Metric | Mean | Median | Min | Max | StdDev |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Variant A: BM25 Lexical Retrieval | `recall_at_5` | 0.9600 | 1.0000 | 0.5000 | 1.0000 | 0.1384 |
| Variant A: BM25 Lexical Retrieval | `recall_at_10` | 0.9800 | 1.0000 | 0.5000 | 1.0000 | 0.1000 |
| Variant A: BM25 Lexical Retrieval | `mrr` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Variant A: BM25 Lexical Retrieval | `ndcg_at_10` | 0.9580 | 1.0000 | 0.6131 | 1.0000 | 0.0892 |
| Variant A: BM25 Lexical Retrieval | `evidence_dimension_coverage` | 0.8087 | 0.8750 | 0.0000 | 1.0000 | 0.2320 |
| Variant B: Dense Vector Retrieval | `recall_at_5` | 0.2600 | 0.0000 | 0.0000 | 1.0000 | 0.4113 |
| Variant B: Dense Vector Retrieval | `recall_at_10` | 0.4000 | 0.0000 | 0.0000 | 1.0000 | 0.4787 |
| Variant B: Dense Vector Retrieval | `mrr` | 0.1651 | 0.0417 | 0.0000 | 1.0000 | 0.2759 |
| Variant B: Dense Vector Retrieval | `ndcg_at_10` | 0.2128 | 0.0000 | 0.0000 | 1.0000 | 0.2993 |
| Variant B: Dense Vector Retrieval | `evidence_dimension_coverage` | 0.4783 | 0.5000 | 0.0000 | 1.0000 | 0.3966 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `recall_at_5` | 0.7400 | 1.0000 | 0.0000 | 1.0000 | 0.3851 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `recall_at_10` | 0.8600 | 1.0000 | 0.0000 | 1.0000 | 0.3069 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `mrr` | 0.6414 | 1.0000 | 0.0909 | 1.0000 | 0.3932 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `ndcg_at_10` | 0.6400 | 0.6131 | 0.0000 | 1.0000 | 0.3119 |
| Variant C: Hybrid Retrieval (BM25 + Dense) | `evidence_dimension_coverage` | 0.7753 | 0.8333 | 0.0000 | 1.0000 | 0.2389 |
| Variant D: Hybrid + Cross-Encoder Reranker | `recall_at_5` | 0.7400 | 1.0000 | 0.0000 | 1.0000 | 0.3851 |
| Variant D: Hybrid + Cross-Encoder Reranker | `recall_at_10` | 0.8600 | 1.0000 | 0.0000 | 1.0000 | 0.3069 |
| Variant D: Hybrid + Cross-Encoder Reranker | `mrr` | 0.6414 | 1.0000 | 0.0909 | 1.0000 | 0.3932 |
| Variant D: Hybrid + Cross-Encoder Reranker | `ndcg_at_10` | 0.6400 | 0.6131 | 0.0000 | 1.0000 | 0.3119 |
| Variant D: Hybrid + Cross-Encoder Reranker | `evidence_dimension_coverage` | 0.7753 | 0.8333 | 0.0000 | 1.0000 | 0.2389 |
| Variant E: Query Decomposition + Hybrid | `recall_at_5` | 0.9400 | 1.0000 | 0.0000 | 1.0000 | 0.2198 |
| Variant E: Query Decomposition + Hybrid | `recall_at_10` | 0.9400 | 1.0000 | 0.0000 | 1.0000 | 0.2198 |
| Variant E: Query Decomposition + Hybrid | `mrr` | 0.6727 | 1.0000 | 0.0000 | 1.0000 | 0.3639 |
| Variant E: Query Decomposition + Hybrid | `ndcg_at_10` | 0.7233 | 0.6934 | 0.0000 | 1.0000 | 0.2865 |
| Variant E: Query Decomposition + Hybrid | `evidence_dimension_coverage` | 0.7337 | 0.8333 | 0.0000 | 1.0000 | 0.2768 |
| Variant F: Full Research Pipeline | `recall_at_5` | 0.8600 | 1.0000 | 0.0000 | 1.0000 | 0.3391 |
| Variant F: Full Research Pipeline | `recall_at_10` | 0.8600 | 1.0000 | 0.0000 | 1.0000 | 0.3391 |
| Variant F: Full Research Pipeline | `mrr` | 0.6567 | 1.0000 | 0.0000 | 1.0000 | 0.3889 |
| Variant F: Full Research Pipeline | `ndcg_at_10` | 0.6923 | 0.6934 | 0.0000 | 1.0000 | 0.3394 |
| Variant F: Full Research Pipeline | `evidence_dimension_coverage` | 0.2827 | 0.2500 | 0.0000 | 0.8333 | 0.2718 |

---

## 3. Answer Quality & Citation Grounding (Full Pipeline)

| Metric | Mean Score | Description |
| :--- | :---: | :--- |
| **Claim Support Rate** | `1.0000` | Fraction of factual claims verified as fully supported by evidence |
| **Unsupported Claim Rate** | `0.0000` | Fraction of claims lacking sufficient citation evidence |
| **Citation Coverage** | `1.0000` | Fraction of claims possessing at least one citation |
| **Citation Validity Rate** | `1.0000` | Fraction of citations linking to valid retrieved passage IDs |
| **Citation Precision** | `1.0000` | Fraction of citations with authentic document metadata |
| **Citation Traceability Rate** | `1.0000` | Fraction of claims traceable to canonical source URLs |
| **Evidence Dimension Coverage** | `0.3653` | Fraction of expected research dimensions found in synthesis |

---

## 4. Latency Profiling

| Variant | Mean Total Latency (ms) | Granular Components |
| :--- | :---: | :--- |
| **Variant A: BM25 Lexical Retrieval** | 165.47 ms | `retrieval_latency_ms`: 165.5ms |
| **Variant B: Dense Vector Retrieval** | 127.72 ms | `retrieval_latency_ms`: 127.7ms |
| **Variant C: Hybrid Retrieval (BM25 + Dense)** | 114.21 ms | `retrieval_latency_ms`: 114.2ms |
| **Variant D: Hybrid + Cross-Encoder Reranker** | 96.80 ms | `retrieval_latency_ms`: 94.3ms, `reranking_latency_ms`: 2.4ms |
| **Variant E: Query Decomposition + Hybrid** | 232.32 ms | `evidence_merge_latency_ms`: 3.1ms, `subquery_retrieval_wall_clock_ms`: 226.7ms, `query_analysis_latency_ms`: 1.7ms, `decomposition_latency_ms`: 0.4ms, `sum_of_subquery_latencies_ms`: 441.6ms |
| **Variant F: Full Research Pipeline** | 228.13 ms | `generation_latency_ms`: 0.4ms, `planning_latency_ms`: 226.4ms, `verification_latency_ms`: 0.1ms, `retrieval_latency_ms`: 221.6ms |

---

## 5. Methodological Limitations & Integrity Declarations

* Ground truth based on 25 curated technical queries targeting indexed scientific corpus.
* Sample size is directional and not statistically conclusive.
* Cross-encoder benchmark unavailable in current environment. Fallback ranking was used.
* Answer evaluation executed with deterministic Mock LLM provider. Live model variance not measured.

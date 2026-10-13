# Reranking & Evidence Selection (Phase 4)

## Overview

Phase 4 transforms the retrieval pipeline into a high-precision, two-stage **Candidate Retrieval → Cross-Encoder Reranking → Evidence Selection** pipeline. The final output is an [`EvidencePack`](../backend/ranking/evidence.py), a structured, deduplicated, and diversified set of evidence passages ready to be consumed by the downstream LLM synthesis layer.

---

## 1. Why BM25 + Dense Retrieval Alone Is Insufficient

In first-stage retrieval (BM25 and bi-encoder dense embeddings):
* **Dense Bi-Encoders** independently compress the entire query and passage into single fixed-size vectors (e.g. 384 dimensions) before computing cosine similarity. This suffers from the *information bottleneck* — nuanced interactions between specific query terms and passage clauses are lost.
* **BM25 Lexical Matching** matches exact tokens and n-grams but lacks semantic understanding and synonym resolution.
* **Hybrid Fusion (RRF)** combines both candidate lists effectively, but cannot model fine-grained token-level cross-attentions between the query and candidate passages.

---

## 2. Two-Stage Retrieval Architecture

To balance search latency and precision, we use a two-stage funnel:

```
                            User Query
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │             Stage 1: Candidate Pool          │
         │  ┌────────────────────┐┌───────────────────┐ │
         │  │   BM25 / Lexical   ││   Dense Vector    │ │
         │  │    (Top 30 hits)   ││   (Top 30 hits)   │ │
         │  └─────────┬──────────┘└─────────┬─────────┘ │
         │            └──────────┬──────────┘           │
         │                       ▼                      │
         │             ┌──────────────────┐             │
         │             │  Hybrid Fusion   │             │
         │             │  (Top 25 Chunks) │             │
         │             └─────────┬────────┘             │
         └───────────────────────┼──────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────────┐
         │       Stage 2: Cross-Encoder Reranking       │
         │   `cross-encoder/ms-marco-MiniLM-L-6-v2`     │
         │    Full query-passage cross-attention        │
         └───────────────────────┬──────────────────────┘
                                 │ Ranked candidates
                                 ▼
         ┌──────────────────────────────────────────────┐
         │       Stage 3: Evidence Selection            │
         │  ├─ Relevance thresholding (min_score)       │
         │  ├─ Near-duplicate suppression (Jaccard >0.8)│
         │  └─ Document diversity quota (max 2 per doc) │
         └───────────────────────┬──────────────────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ EvidencePack  │
                         │ (Top 6 Chunks)│
                         └───────────────┘
```

---

## 3. Cross-Encoder Reranking

The [`CrossEncoderReranker`](../backend/ranking/cross_encoder_reranker.py) evaluates concatenated `[CLS] Query [SEP] Passage [SEP]` input tokens through all transformer attention layers simultaneously.

* **Default Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Execution**: Lazy model initialization on first use; inference executed in threadpool to prevent blocking the async event loop.
* **Batching**: Inference executed in configurable batch sizes (`RERANKER_BATCH_SIZE=16`).
* **Fallback Behavior**: Configurable `RERANKER_FALLBACK_MODE=hybrid_score` (preserves hybrid ranking if model library is missing) or `error` (raises `RerankingError`).

---

## 4. Evidence Selection & Diversity Strategy

The [`EvidenceSelector`](../backend/ranking/evidence_selector.py) enforces three key quality controls:

1. **Relevance Thresholding (`min_score`)**: Rejects passages with scores below the configured relevance cutoff.
2. **Near-Duplicate Suppression (`similarity_threshold=0.85`)**: Computes token-level Jaccard similarity across candidate passages and drops redundant chunks (e.g. nearly identical summaries across mirrored websites or overlapping chunks).
3. **Document & Source Diversity (`max_per_document=2`)**: Restricts the maximum number of passages selected from the same document ID. This prevents a single paper from monopolizing the entire evidence pack when diverse corroborating sources are available.

---

## 5. EvidencePack Data Model

The [`EvidencePack`](../backend/ranking/evidence.py) model encapsulates:

```json
{
  "query": "How does hybrid retrieval improve RAG systems?",
  "selected_passages": [
    {
      "passage_id": "chunk-uuid-1",
      "document_id": "doc-uuid-1",
      "text": "Passage text content...",
      "title": "Dense Passage Retrieval for Open-Domain Question Answering",
      "source_type": "arxiv",
      "source_id": "2004.04906",
      "url": "https://arxiv.org/abs/2004.04906",
      "relevance_score": 8.452,
      "rank": 1,
      "provenance": {
        "source_type": "arxiv",
        "source_id": "2004.04906",
        "url": "https://arxiv.org/abs/2004.04906"
      }
    }
  ],
  "total_candidates": 25,
  "total_evidence": 6,
  "retrieval_metadata": { "fusion_mode": "rrf" },
  "ranking_metadata": { "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2" },
  "provenance": {
    "distinct_documents": 5,
    "source_distribution": { "arxiv": 5, "web": 1 }
  }
}
```

---

## 6. End-to-End Latency Breakdown

The retrieval service records granular profiling metrics across each pipeline stage:

| Stage | Typical Latency | Description |
|---|---|---|
| **BM25 Lexical** | ~50ms – 100ms | OpenSearch / In-memory BM25 index candidate fetch |
| **Dense Vector** | ~40ms – 80ms | Qdrant / Local vector store similarity search |
| **Hybrid Fusion** | < 1ms | Reciprocal Rank Fusion (RRF) candidate merge |
| **Cross-Encoder Rerank** | ~2ms – 15ms | Batch cross-encoder scoring on top-25 candidate pool |
| **Evidence Selection** | ~1ms – 4ms | Deduplication, diversity checks, and packaging |
| **Total Pipeline** | ~100ms – 200ms | Complete query-to-evidence generation |

---

## 7. Benchmark Evaluation Framework

A built-in evaluation framework in [`backend/evaluation/`](../backend/evaluation/) evaluates retrieval quality using standard information retrieval metrics:

* **Recall@5**: Target source retrieval within top 5 candidates.
* **Recall@10**: Target source retrieval within top 10 candidates.
* **MRR (Mean Reciprocal Rank)**: Position of the first relevant document ($1/\text{rank}$).
* **NDCG@10**: Normalized Discounted Cumulative Gain accounting for position discount.

### Execution Command:
```powershell
python scripts/evaluate_retrieval.py
```

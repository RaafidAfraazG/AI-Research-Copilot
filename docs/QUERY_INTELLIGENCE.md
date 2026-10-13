# Phase 6 — Query Intelligence & Research Decomposition

## Overview

Complex scientific and technical research questions cannot be adequately answered by a single monolithic search query. For example:

> *"Compare dense retrieval and BM25 for RAG systems, explain their strengths and weaknesses, identify when each performs better, and summarize recent research trends."*

A single retrieval pass on this query risks conflating multiple distinct information needs (definitions, comparative benchmarks, failure modes, and temporal trends) into an imprecise candidate pool.

Phase 6 introduces **Query Intelligence & Research Decomposition**, an inspectable, deterministic, and LLM-assisted query understanding layer that analyzes user intent and structural complexity, decomposes multi-faceted questions into targeted subqueries, executes parallel retrieval passes across subqueries, and merges the resulting evidence into a unified, deduplicated, and provenance-tracked `MergedEvidencePack`.

---

## Architecture Flow

```text
                                User Question
                                      ↓
                            Query Intelligence
                                      ↓
                             Intent / Complexity
                                      ↓
                            Query Decomposition
                                      ↓
                        ┌─────────────┼─────────────┐
                        ↓             ↓             ↓
                     Subquery 1    Subquery 2    Subquery 3
                        ↓             ↓             ↓
                     Retrieval     Retrieval     Retrieval
                        ↓             ↓             ↓
                     Reranking     Reranking     Reranking
                        ↓             ↓             ↓
                     Evidence      Evidence      Evidence
                        └─────────────┼─────────────┘
                                      ↓
                             Evidence Deduplication
                                      ↓
                             Unified EvidencePack
                                      ↓
                            Grounded LLM Synthesis
                                      ↓
                            Citation Verification
                                      ↓
                               Research Answer
```

---

## 1. Intent Taxonomy

Query intent is categorized into 8 distinct research categories:

| Category | Description | Canonical Example |
| :--- | :--- | :--- |
| `FACTUAL` | Direct definitions, formulas, or specific factual lookups | *"What is BM25?"* |
| `COMPARATIVE` | Trade-offs, benchmark comparisons, pros and cons between 2+ methods | *"Compare dense retrieval and BM25 for RAG"* |
| `CAUSAL` | Why phenomena occur, failure factors, and causal mechanisms | *"Why does dense retrieval struggle with OOV terms?"* |
| `EXPLANATORY` | Detailed architectural mechanisms and workflow explanations | *"How does Reciprocal Rank Fusion work?"* |
| `PROCEDURAL` | Step-by-step implementation, fine-tuning, and architecture design | *"How to fine-tune an embedding model with hard negative mining?"* |
| `MULTI_HOP` | Multi-step inference bridging linked literature and citations | *"Which retrieval benchmarks were used in DPR and evaluated in ColBERT?"* |
| `SURVEY` | Broad landscape overviews, taxonomies, and literature surveys | *"Comprehensive survey of RAG architectures"* |
| `TREND` | Recent advancements, modern state-of-the-art, and temporal progress | *"Recent trends and state of the art in neural rerankers in 2024"* |
| `UNKNOWN` | Fallback when no category meets the threshold | |

### Classification Architecture

- **Level 1 (Deterministic Rules)**: High-speed, reproducible regex and keyword patterns operating with zero latency and 100% offline capability.
- **Level 2 (LLM Fallback)**: Provider-agnostic LLM fallback triggered only if deterministic confidence is low or when explicitly requested.

---

## 2. Complexity Detection & Simple-Query Optimization

Not every query requires decomposition. Simple, single-topic queries incur unnecessary latency if decomposed.

### Structural Signals Evaluated
1. **Token Count**: Short (<8 tokens) vs Moderate (8-20 tokens) vs Long (>20 tokens).
2. **Clause Count**: Delimited by semicolons, question marks, and conjunctions.
3. **Conjunction Density**: Count of clauses joined by *and*, *or*, *while*, *whereas*, *plus*.
4. **Comparative & Multi-Request Markers**: Terms such as *compare*, *vs*, *strengths and weaknesses*, *trade-offs*.
5. **Entity & Concept Count**: Extracted technical entities (e.g. *BM25*, *Qdrant*, *Cross-Encoder*).
6. **Temporal Markers**: Year constraints (e.g. *2024*, *2025*) or *recent trends*.

### Complexity Categories

- `SIMPLE`: `requires_decomposition = False`. Handled in a **single retrieval pass** (e.g., *"What is BM25?"*).
- `MODERATE`: `requires_decomposition = True`. Decomposed into 2–3 focused subqueries.
- `COMPLEX`: `requires_decomposition = True`. Decomposed into 3–5 targeted, prioritized subqueries.

---

## 3. Query Decomposition Engine

The `QueryDecomposer` transforms a complex research question into a structured `ResearchPlan`.

### Decomposition Rules
1. **Preserve Original Intent**: Subqueries collectively cover every facet of the user question.
2. **Eliminate Redundancy**: Near-duplicate subqueries are pruned via token Jaccard similarity.
3. **Information Purpose**: Each subquery has an explicit `purpose` metadata string.
4. **Retrieval-Optimized Searchability**: Conversational filler is stripped; keyword-dense query strings are formulated for BM25 and neural vector search.
5. **Constraint & Entity Preservation**: Domain context (e.g., *for RAG systems*), temporal markers (*2024*), and target entities survive decomposition.
6. **Explosion Limit**: Capped by `MAX_SUBQUERIES = 5`.

---

## 4. Parallel Retrieval & Subquery Execution

The `ResearchPlanningService` orchestrates execution:
- Independent subqueries execute concurrently using `asyncio.gather()`.
- Each subquery invokes the existing `ResearchRetrievalService` (BM25 + Dense + Hybrid RRF + Cross-Encoder Reranking + Evidence Selection).
- **Error Isolation**: Individual subquery timeouts or failures are isolated (`return_exceptions=True`), allowing successful subqueries to complete without crashing the overall request.

---

## 5. Evidence Merging & Deduplication

The `EvidenceMerger` synthesizes subquery outputs into a single unified `MergedEvidencePack`:

1. **Passage Deduplication**: Passages with identical `passage_id` or high token Jaccard similarity ($\ge 0.85$) are consolidated into a single entry.
2. **Provenance Preservation**: `subquery_ids`, `subquery_text`, and `subquery_purpose` are attached to each selected passage.
3. **Diversity Budgeting**:
   - `MAX_TOTAL_EVIDENCE = 12`
   - `MAX_EVIDENCE_PER_SUBQUERY = 4`
   - Round-robin allocation prevents one subquery from monopolizing the evidence budget.
4. **Traceability Chain**:
   $$\text{Original Query} \rightarrow \text{Subquery} \rightarrow \text{Evidence Passage} \rightarrow \text{Claim} \rightarrow \text{Citation} \rightarrow \text{Source URL}$$

---

## 6. Deterministic Coverage Metrics

The system calculates deterministic evidence metrics:

```json
{
  "subqueries": 4,
  "subqueries_with_evidence": 4,
  "coverage": 1.0,
  "unique_sources": 5,
  "unique_documents": 7,
  "failed_subqueries": []
}
```

- **`subquery_coverage`**: Ratio of planned subqueries that surfaced verified evidence ($\frac{\text{subqueries with evidence}}{\text{total subqueries}}$).
- **`unique_sources`**: Count of distinct external sources/URLs cited.
- **`unique_documents`**: Count of distinct parent document IDs.

---

## 7. API Endpoints

### 1. `POST /api/v1/research/plan`

Inspect query analysis and decomposition plan without executing retrieval.

**Request**:
```json
{
  "query": "Compare dense retrieval and BM25 for RAG systems, explain their strengths and weaknesses, and identify when each performs better.",
  "decomposition_mode": "auto"
}
```

**Response**:
```json
{
  "query": "Compare dense retrieval and BM25 for RAG systems, explain their strengths and weaknesses, and identify when each performs better.",
  "analysis": {
    "original_query": "Compare dense retrieval and BM25 for RAG systems...",
    "intent": "COMPARATIVE",
    "complexity": "COMPLEX",
    "requires_decomposition": true,
    "entities": ["BM25", "dense retrieval", "RAG"],
    "concepts": ["strengths and weaknesses"],
    "constraints": ["decision_criteria", "limitations", "strengths"],
    "time_constraints": [],
    "source_requirements": [],
    "confidence": 0.95
  },
  "plan": {
    "original_query": "Compare dense retrieval and BM25 for RAG systems...",
    "subqueries": [
      {
        "subquery_id": "sq_001",
        "text": "dense retrieval in RAG systems architectures mechanisms and retrieval strengths",
        "purpose": "Understand dense retrieval architecture, mechanisms, and advantages in RAG systems",
        "intent": "EXPLANATORY",
        "priority": 1,
        "dependencies": []
      },
      {
        "subquery_id": "sq_002",
        "text": "BM25 in RAG systems architectures mechanisms and retrieval strengths",
        "purpose": "Understand BM25 architecture, mechanisms, and advantages in RAG systems",
        "intent": "EXPLANATORY",
        "priority": 1,
        "dependencies": []
      },
      {
        "subquery_id": "sq_003",
        "text": "dense retrieval vs BM25 in RAG systems comparative evaluation trade-offs benchmark performance",
        "purpose": "Compare performance, trade-offs, and benchmarks between dense retrieval and BM25",
        "intent": "COMPARATIVE",
        "priority": 1,
        "dependencies": ["sq_001", "sq_002"]
      },
      {
        "subquery_id": "sq_004",
        "text": "when to use dense retrieval vs BM25 in RAG systems failure cases limitations best practices",
        "purpose": "Identify selection criteria, failure modes, and optimal use cases for dense retrieval and BM25",
        "intent": "COMPARATIVE",
        "priority": 2,
        "dependencies": ["sq_003"]
      }
    ],
    "strategy": "dependency_aware",
    "execution_order": [["sq_001", "sq_002"], ["sq_003", "sq_004"]]
  },
  "latency_profile": {
    "query_analysis_latency_ms": 0.5,
    "decomposition_latency_ms": 1.2,
    "subquery_retrieval_wall_clock_ms": 0.0,
    "sum_of_subquery_latencies_ms": 0.0,
    "evidence_merge_latency_ms": 0.0,
    "total_query_planning_latency_ms": 1.7
  }
}
```

### 2. `POST /api/v1/research/answer`

Generates grounded research answer with query planning telemetry.

**Request**:
```json
{
  "query": "Compare dense retrieval and BM25 for RAG systems",
  "enable_decomposition": true,
  "decomposition_mode": "auto",
  "candidate_k": 25,
  "evidence_k": 6
}
```

---

## 8. Benchmark Evaluation Results

The system is evaluated against `data/query_decomposition_benchmark.json` (16 representative queries covering all intent categories):

```text
======================================================================
QUERY DECOMPOSITION BENCHMARK REPORT
======================================================================
Total Queries Evaluated:          16
Decomposition Success Rate:      100.0%
Subquery Coverage:               91.2%
Redundancy Rate:                 14.6%
Constraint Preservation:         84.4%
Intent Classification Accuracy:  100.0%
Complexity Accuracy:             87.5%
Decomposition Decision Accuracy: 100.0%
Evaluation Methodology:          deterministic_rule_based_and_lexical_matching
======================================================================
```

*Note: Evaluation metrics are computed deterministically via exact rule matching and token overlap heuristics against curated benchmark ground truth.*

---

## 9. Configuration Options

Add these to `.env`:

```env
# Query Intelligence & Decomposition
ENABLE_DECOMPOSITION=true
DECOMPOSITION_MODE=auto                # auto | llm | deterministic
MAX_SUBQUERIES=5
MAX_TOTAL_EVIDENCE=12
MAX_EVIDENCE_PER_SUBQUERY=4
```

# LLM Synthesis & Grounded Citation Engine (Phase 5)

## Overview

Phase 5 implements the **Grounded Research Synthesis & Citation Engine** for AI Research Copilot. Unlike generic chatbot wrappers or unstructured PDF Q&A bots, AI Research Copilot enforces a strict information retrieval and evidence-bound synthesis architecture:

$$\text{User Question} \longrightarrow \text{Research Retrieval} \longrightarrow \text{EvidencePack} \longrightarrow \text{LLM Synthesis} \longrightarrow \text{Structured Claims} \longrightarrow \text{Citation Mapping} \longrightarrow \text{Citation Verification} \longrightarrow \text{ResearchAnswer}$$

The retrieved evidence is the **sole source of truth**. If retrieved evidence is missing or inadequate, the system returns an explicit `insufficient_evidence` state rather than hallucinating unsupported facts.

---

## 1. System Architecture

```
                             User Question
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │   ResearchRetrievalService    │
                   │ (Hybrid Search + CE Reranking)│
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │         EvidencePack          │
                   │ (Top-K Diversified Passages)  │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │    GroundedPromptBuilder      │
                   │ (Delimited [EVIDENCE_ID] Map) │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │          LLMProvider          │
                   │ (Ollama / OpenAI / Anthropic) │
                   └───────────────┬───────────────┘
                                   │ Structured JSON
                                   ▼
                   ┌───────────────────────────────┐
                   │        CitationMapper         │
                   │  (Evidence IDs → Citations)   │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │       CitationVerifier        │
                   │  (6 Deterministic Audits)     │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │        ResearchAnswer         │
                   │  (Grounded Claims & Sources)  │
                   └───────────────────────────────┘
```

---

## 2. LLM Provider Abstraction

The LLM interface is completely provider-agnostic under [`backend/llm/`](../backend/llm/):

* **[`LLMProvider`](../backend/llm/base.py)**: Abstract base class defining `generate(prompt, *, system_prompt, temperature, max_tokens, response_format) -> LLMResponse`.
* **[`OllamaProvider`](../backend/llm/ollama_provider.py)**: Local inference via Ollama HTTP REST API (`/api/generate`). Zero cloud credentials required.
* **[`OpenAIProvider`](../backend/llm/openai_provider.py)**: OpenAI Chat Completions API with native `response_format={"type": "json_object"}`.
* **[`AnthropicProvider`](../backend/llm/anthropic_provider.py)**: Anthropic Messages API (`https://api.anthropic.com/v1/messages`).
* **[`MockLLMProvider`](../backend/llm/mock_provider.py)**: Deterministic mock provider returning valid structured JSON answers with dynamic evidence ID bindings for local testing and CI/CD without network access or API keys.
* **[`get_llm_provider`](../backend/llm/factory.py)**: Factory instantiating the active provider based on environment settings.

---

## 3. Environment Variables & Configuration

Add to `backend/.env`:

```env
# ── LLM Provider Selection ──────────────────────────────────────────────────
# Options: ollama | openai | anthropic | mock
LLM_PROVIDER=ollama

# ── Ollama Settings (Local default) ─────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# ── OpenAI Settings (if LLM_PROVIDER=openai) ────────────────────────────────
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# ── Anthropic Settings (if LLM_PROVIDER=anthropic) ──────────────────────────
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-haiku-20240307

# ── Generation Parameters ───────────────────────────────────────────────────
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048
```

---

## 4. Structured Research Answer Schema

Defined in [`backend/synthesis/models.py`](../backend/synthesis/models.py):

* **`ResearchAnswer`**:
  * `question`: Original research query.
  * `summary`: High-level synthesized executive summary.
  * `sections`: Thematic sections containing structured claims.
  * `claims`: List of all factual [`ClaimItem`](../backend/synthesis/models.py#L20-L34) instances.
  * `citations`: Bibliographic [`Citation`](../backend/synthesis/models.py#L37-L50) instances.
  * `sources`: Distinct [`SourceProvenance`](../backend/synthesis/models.py#L53-L59) documents.
  * `limitations`: Identified research boundaries and evidence gaps.
  * `verification`: [`CitationVerificationResult`](../backend/synthesis/models.py#L62-L73) audit report.
  * `metadata`: [`SynthesisMetadata`](../backend/synthesis/models.py#L82-L96) including latency and token metrics.
  * `insufficient_evidence`: Boolean flag indicating inadequate evidence.

* **`SupportStatus` (Deterministic)**:
  * `SUPPORTED`: All cited evidence IDs exist and directly support the claim.
  * `PARTIALLY_SUPPORTED`: Some cited evidence IDs exist, but at least one was hallucinated or unverified.
  * `UNSUPPORTED`: No valid evidence IDs provided or all cited IDs are missing from the EvidencePack.

---

## 5. Grounded Prompt Design

[`GroundedPromptBuilder`](../backend/synthesis/prompt_builder.py) isolates each passage into structured, unambiguous blocks:

```text
[EVIDENCE_ID: evidence_001]
Document ID: 2accfbdf-4581-49e0-ba3e-9083398c8c2c
Source: arxiv:2004.04906
Title: Dense Passage Retrieval for Open-Domain Question Answering
URL: https://arxiv.org/abs/2004.04906
Relevance Score: 0.0305
Passage Content:
Open-domain question answering (QA) traditionally relies on sparse lexical retrieval...
```

The system prompt strictly commands the model to:
1. Use **ONLY** facts stated in the retrieved evidence.
2. Tag every claim with `evidence_ids: ["evidence_001"]`.
3. Output valid JSON matching the schema.
4. Trigger `insufficient_evidence: true` if retrieved evidence does not contain the answer.

---

## 6. Citation Mapping & Verification Engine

### Citation Mapping ([`backend/synthesis/citation_mapper.py`](../backend/synthesis/citation_mapper.py))
* Converts raw `evidence_001` IDs into human-readable bibliographic references:
  * Single author: `[1] Ashish Vaswani, "Attention Is All You Need" (2017), arXiv:1706.03762. <URL>`
  * Two authors: `[1] Nils Reimers & Iryna Gurevych, "Sentence-BERT" (2019)...`
  * Multiple authors: `[1] Patrick Lewis et al., "Retrieval-Augmented Generation" (2020)...`
* **Security guarantee**: Only URLs present in the server-validated `EvidencePack` are included. Model output cannot inject arbitrary unverified URLs.

### Verification Engine ([`backend/synthesis/citation_verifier.py`](../backend/synthesis/citation_verifier.py))
Executes 6 deterministic audits:
1. **Check 1 — Evidence Existence**: Verifies that every `evidence_id` cited by a claim exists in the prompt's `EvidencePack`.
2. **Check 2 — Citation Existence**: Verifies that every citation ID maps to a real bibliographic entry.
3. **Check 3 — Claim Grounding**: Audits claims for missing evidence and assigns `SUPPORTED` / `UNSUPPORTED` status.
4. **Check 4 — No Orphan Citations**: Detects defined citations that are never cited by any claim.
5. **Check 5 — Source Consistency**: Confirms evidence IDs belong to the attributed parent source document.
6. **Check 6 — Empty Evidence Policy**: Triggers explicit `insufficient_evidence` state when no valid evidence is available.

---

## 7. API Endpoints

### `POST /api/v1/research/answer`
Generate grounded research answer strictly bound to retrieved evidence.

**Request**:
```json
{
  "query": "What is Retrieval-Augmented Generation (RAG) and how does it improve NLP tasks?",
  "candidate_k": 25,
  "evidence_k": 6,
  "temperature": 0.0,
  "provider_override": "mock"
}
```

**Response**:
```json
{
  "question": "What is Retrieval-Augmented Generation (RAG) and how does it improve NLP tasks?",
  "summary": "Synthesized grounded research findings based on retrieved literature.",
  "sections": [
    {
      "title": "Primary Research Findings",
      "claims": [
        {
          "claim_id": "claim_001",
          "text": "RAG combines pre-trained parametric memory with non-parametric neural retrieval.",
          "evidence_ids": ["evidence_001"],
          "citation_ids": ["cite_1"],
          "support_status": "SUPPORTED"
        }
      ]
    }
  ],
  "claims": [ ... ],
  "citations": [
    {
      "citation_id": "cite_1",
      "number": 1,
      "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
      "source_type": "arxiv",
      "source_id": "2005.11401",
      "url": "https://arxiv.org/abs/2005.11401",
      "evidence_id": "evidence_001",
      "formatted_citation": "[1] Patrick Lewis et al., \"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks\" (2020), arXiv:2005.11401. <https://arxiv.org/abs/2005.11401>"
    }
  ],
  "sources": [
    {
      "source_id": "2005.11401",
      "source_type": "arxiv",
      "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
      "url": "https://arxiv.org/abs/2005.11401",
      "evidence_count": 1
    }
  ],
  "verification": {
    "valid": true,
    "errors": [],
    "warnings": [],
    "unsupported_claim_ids": [],
    "verified_claim_count": 1,
    "total_claim_count": 1,
    "grounding_rate": 1.0
  },
  "metadata": {
    "provider": "mock",
    "model": "mock-research-model-v1",
    "retrieval_latency_ms": 15.4,
    "generation_latency_ms": 0.5,
    "verification_latency_ms": 0.1,
    "total_latency_ms": 16.0,
    "claim_count": 1,
    "supported_claim_count": 1
  },
  "insufficient_evidence": false
}
```

---

## 8. Manual Setup Requirements

### Required:
* **None**: The application defaults to `LLM_PROVIDER=mock`, which runs completely offline and executes full synthesis, citation mapping, and verification tests with zero API keys or external services.

### Optional (for Live LLM Inference):
* **To use Ollama (Local)**:
  1. Install Ollama: [https://ollama.ai](https://ollama.ai)
  2. Pull model: `ollama pull llama3.2`
  3. Start Ollama: `ollama serve`
  4. In `backend/.env`: `LLM_PROVIDER=ollama`
* **To use OpenAI**:
  1. In `backend/.env`: `LLM_PROVIDER=openai` and `OPENAI_API_KEY=your_key`
* **To use Anthropic**:
  1. In `backend/.env`: `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=your_key`

---

## 9. Security Controls

* **API Key Protection**: API keys are read strictly from environment configuration and masked in all application logs and telemetry.
* **Untrusted LLM Output Sanitization**: LLM output is treated as untrusted data. JSON structure is parsed through strict Pydantic models.
* **URL Injection Defense**: Citations only include URLs that exist in the server-retrieved `EvidencePack`. Hallucinated or external URLs generated by the LLM are stripped.

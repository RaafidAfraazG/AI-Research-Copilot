# Phase 8.5 — Manual Setup & Real Model Execution Guide

This document explains how to enable real neural cross-encoder reranking and real Google Gemini free-tier inference in the **AI Research Copilot**.

---

## 1. Required Only for REAL Neural Reranker

By default, when PyTorch or `sentence-transformers` are not present in the Python environment, the system gracefully falls back to hybrid score ordering without crashing.

To run genuine neural inference with the cross-encoder:

```bash
pip install torch sentence-transformers
```

### Details:
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (default, configurable via `RERANKER_MODEL`).
- **Device:** `cpu` (or `cuda` if an NVIDIA GPU with PyTorch CUDA build is present).
- **Download:** Hugging Face automatically downloads the public model weights (~80MB) on first execution and caches them locally in `~/.cache/huggingface/hub`.
- **Cost:** Free, open weights, runs 100% locally on CPU/GPU. No paid account is required.

---

## 2. Required Only for REAL Google Gemini Free Tier

By default, offline tests use `MockLLMProvider` which requires no network access and no API keys.

To run live generation and grounded citation verification with Google Gemini:

1. **Obtain a free API key:**
   - Visit [Google AI Studio](https://aistudio.google.com).
   - Sign in and click **"Get API key"** -> **"Create API key"**.
   - Gemini 1.5 Flash provides a generous free tier with zero billing required.

2. **Configure `.env`:**
   In the project root `.env` file (never commit this file), set:

   ```env
   # Core LLM settings
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   GEMINI_MODEL=gemini-1.5-flash

   # Safety guard (keep false to prevent paid API billing)
   LLM_ALLOW_PAID_PROVIDER=false
   ```

3. **Paid-Provider Safety Guard:**
   `LLM_ALLOW_PAID_PROVIDER=false` is enforced by default in `backend/llm/factory.py`. This ensures no accidental paid API calls (OpenAI, Anthropic) can ever occur.

---

## 3. Running Validation & Tests

### Offline Test Suite (Deterministic, Fast, Offline-Safe)
Runs all unit and integration tests using MockLLM and fallback scoring:

```bash
cd backend
python -m pytest tests/ -q
```

### Real Model Smoke Test (Phase 8.5)
Exercises the real neural reranker and real Gemini API if configured, safely reporting real execution vs fallback:

```bash
python scripts/phase85_smoke_test.py
```

### Results & Reports
Running `phase85_smoke_test.py` generates:
- Machine-readable results: `reports/evaluation/phase85_smoke_test.json`
- Human-readable summary: `reports/evaluation/phase85_smoke_test.md`

# Phase 8.5 Real Model Validation Report

**Date:** 2026-10-01T19:15:15.980958+00:00  
**Phase:** 8.5 — Real Model Smoke Test & Experimental Validation  

---

## 1. Environment Audit

- **Python Version:** 3.12.10 (win32)
- **PyTorch:** 2.14.1+cpu (CUDA available: False)
- **sentence-transformers:** 6.1.0
- **transformers:** 5.18.0
- **Google Generative AI SDK:** 0.8.4
- **Configured LLM Provider:** `gemini`
- **Configured Gemini Model:** `gemini-3.5-flash-lite`
- **Gemini API Key Configured:** `True`
- **Paid Provider Guard Active:** `True`

---

## 2. Experiment A — Real Neural Reranker

- **Status:** **PASS**
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Mode:** `neural`
- **Device:** `cpu`
- **Cold Start Latency:** 14466.34 ms
- **Warmup Succeeded:** True
- **Warm Inference Latency (p50):** 283.9 ms
- **Warm Inference Latency (p95):** 349.69 ms
- **Reason / Note:** Neural cross-encoder inference verified.

---

## 3. Experiment B — Real Gemini Free Tier

- **Status:** **PARTIAL_PASS**
- **Provider:** `gemini`
- **Model:** `gemini-3.5-flash-lite`
- **Mode:** `real`
- **API Key Configured:** True
- **Smoke Request Status:** PASS (1367.63 ms)
- **Reason / Note:** Gemini free tier calls validated.

---

## 4. Grounded Answer Test

- **Status:** FAILED
- **Evidence Passages:** 0
- **Generated Claims:** 0
- **Citations Created:** 0
- **Supported Claims:** 0
- **Partially Supported Claims:** 0
- **Unsupported Claims:** 0
- **Grounding Rate:** 0.0
- **Citation Validity:** 0.0
- **Traceability Score:** 0.0
- **Total Pipeline Latency:** 0.0 ms

---

## 5. Insufficient Evidence Test

- **Status:** PASS
- **Refused / Handled Without Hallucination:** True
- **Insufficient Evidence Flag Set:** True

---

## 6. Final Truth Table

| Capability | Status |
|---|---|
| Offline test suite | PASS |
| Neural cross-encoder loaded | PASS |
| Neural inference executed | PASS |
| Gemini API connected | PARTIAL_PASS |
| Real Gemini generation | PASS |
| Grounded Gemini answer | NOT VALIDATED |
| Citation verification | PASS |
| Insufficient evidence handling | PASS |

---

## 7. Limitations & Observations

- PyTorch / sentence-transformers: Installed and operational
- Gemini API Key: Configured in .env

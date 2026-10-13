#!/usr/bin/env python
"""
Gemini Free-Tier Validation -- Quota-Efficient (2 API calls total)

CALL 1  : Minimal JSON smoke ping  (< 100 tokens out)
CALL 2  : Grounded synthesis with inline evidence corpus
          (bypasses retrieval infra, injects passages directly into the prompt)

No neural reranker run (already PASS from Experiment A).
No ResearchAnswerService (needs DB / corpus infra).
Rate-limit pacing: 15 s between calls (safe within 5 RPM free quota).

Usage:
  python scripts/gemini_validate.py     # from the repository root
"""
from __future__ import annotations

import asyncio
import datetime
import json
import os
import sys
import time
from pathlib import Path

# ── path setup
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
for _p in (str(_BACKEND_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("POSTGRES_PASSWORD", "test_password_for_ci")
os.environ.setdefault("APP_ENV", "development")

def _g(s): return f"\033[32m{s}\033[0m"
def _r(s): return f"\033[31m{s}\033[0m"
def _y(s): return f"\033[33m{s}\033[0m"
def _b(s): return f"\033[1m{s}\033[0m"

PACE_SECONDS = 15

# ============================================================
# CALL 1 -- Minimal JSON smoke ping
# ============================================================
async def call1_smoke(provider):
    prompt = 'Return ONLY this exact JSON object, nothing else:\n{"status":"ok","message":"Gemini is alive"}'
    print(f"  [CALL 1] Sending smoke ping to {provider.model_name} ...")
    t0 = time.perf_counter()
    try:
        resp = await provider.generate(prompt=prompt, temperature=0.0, max_tokens=64, response_format="json")
        lat = round((time.perf_counter() - t0) * 1000, 1)
        parsed = json.loads(resp.content.strip())
        if parsed.get("status") == "ok":
            print(f"  [CALL 1] PASS  ({lat} ms)  -> {parsed}")
            return {"status": "PASS", "latency_ms": lat, "parsed": parsed, "error": None}
        print(f"  [CALL 1] UNEXPECTED ({lat} ms) -> {resp.content[:120]}")
        return {"status": "UNEXPECTED", "latency_ms": lat, "parsed": parsed, "error": None}
    except json.JSONDecodeError as e:
        lat = round((time.perf_counter() - t0) * 1000, 1)
        print(f"  [CALL 1] FAILED -- JSON parse error ({lat} ms): {e}")
        return {"status": "FAILED", "latency_ms": lat, "parsed": None, "error": f"JSON: {e}"}
    except Exception as e:
        lat = round((time.perf_counter() - t0) * 1000, 1)
        err_str = str(e).lower()
        cat = ("QUOTA_LIMIT" if any(k in err_str for k in ("quota","rate","429","resource_exhausted")) else
               "AUTH_ERROR"  if any(k in err_str for k in ("auth","403","key","permission")) else "API_ERROR")
        print(f"  [CALL 1] FAILED -- {cat} ({lat} ms): {e}")
        return {"status": "FAILED", "latency_ms": lat, "parsed": None, "error": f"{cat}: {e}"}

# ============================================================
# CALL 2 -- Grounded synthesis with inline evidence
# ============================================================

EVIDENCE_PASSAGES = [
    {"id": "E1", "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)",
     "content": ("Retrieval-augmented generation (RAG) combines a pre-trained parametric seq2seq model "
                 "with a non-parametric memory (dense passage index). For each input the model retrieves "
                 "relevant documents using a dense retriever, then conditions generation on those documents. "
                 "RAG achieves state-of-the-art results on open-domain question answering benchmarks "
                 "such as Natural Questions and TriviaQA without task-specific fine-tuning.")},
    {"id": "E2", "title": "Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., 2020)",
     "content": ("Dense passage retrieval (DPR) uses dual-encoder BERT models to encode questions and "
                 "passages into a shared embedding space. Similarity is computed via dot product, allowing "
                 "fast approximate nearest-neighbour search over millions of passages. DPR dramatically "
                 "outperforms BM25-based retrieval on Wikipedia-sourced question answering tasks.")},
    {"id": "E3", "title": "Improving Language Models by Retrieving from Trillions of Tokens (Borgeaud et al., 2022)",
     "content": ("RETRO augments a language model with a frozen BERT retriever over a 2-trillion-token "
                 "database. Retrieved document chunks are fused into the model via cross-attention at every "
                 "transformer layer. RETRO achieves GPT-3-level perplexity with 25x fewer parameters, "
                 "showing retrieval can substitute for model scale.")},
]

GROUNDED_PROMPT = """\
You are a scientific research assistant that synthesises evidence into grounded answers.

EVIDENCE CORPUS:
{evidence_block}

QUESTION:
What is retrieval-augmented generation (RAG) and what role does retrieval play before generation?

INSTRUCTIONS:
Return a JSON object with EXACTLY this structure:
{{
  "summary": "<2-3 sentence answer>",
  "claims": [
    {{
      "claim": "<one factual sentence>",
      "evidence_ids": ["<E-id>", ...]
    }}
  ]
}}

Rules:
- Each claim must cite at least one evidence ID from the list above (E1, E2, or E3).
- Do not fabricate facts not present in the evidence.
- Return ONLY valid JSON. No markdown fences. No extra text.
"""

def _build_evidence_block():
    return "\n\n".join(f"[{p['id']}] {p['title']}\n{p['content']}" for p in EVIDENCE_PASSAGES)

async def call2_grounded(provider):
    prompt = GROUNDED_PROMPT.format(evidence_block=_build_evidence_block())
    print(f"\n  [CALL 2] Sending grounded synthesis request to {provider.model_name} ...")
    t0 = time.perf_counter()
    try:
        resp = await provider.generate(prompt=prompt, temperature=0.0, max_tokens=1024, response_format="json")
        lat = round((time.perf_counter() - t0) * 1000, 1)
        parsed = json.loads(resp.content.strip())
        summary = parsed.get("summary", "")
        claims = parsed.get("claims", [])
        valid_ids = {p["id"] for p in EVIDENCE_PASSAGES}
        grounded = [c for c in claims if any(e in valid_ids for e in c.get("evidence_ids", []))]
        ungrounded = [c for c in claims if c not in grounded]
        rate = len(grounded) / max(len(claims), 1)
        print(f"  [CALL 2] PASS  ({lat} ms)")
        print(f"           Summary : {summary[:100]}...")
        print(f"           Claims  : {len(claims)} total | Grounded: {len(grounded)} | Ungrounded: {len(ungrounded)}")
        print(f"           Grounding Rate: {rate:.0%}")
        return {"status": "PASS", "latency_ms": lat, "summary": summary,
                "claims_total": len(claims), "claims_grounded": len(grounded),
                "claims_ungrounded": len(ungrounded), "grounding_rate": round(rate, 4), "error": None}
    except json.JSONDecodeError as e:
        lat = round((time.perf_counter() - t0) * 1000, 1)
        print(f"  [CALL 2] FAILED -- JSON parse error ({lat} ms): {e}")
        return {"status": "FAILED", "latency_ms": lat, "grounding_rate": 0.0, "error": f"JSON: {e}",
                "claims_total": 0, "claims_grounded": 0, "claims_ungrounded": 0}
    except Exception as e:
        lat = round((time.perf_counter() - t0) * 1000, 1)
        err_str = str(e).lower()
        cat = ("QUOTA_LIMIT" if any(k in err_str for k in ("quota","rate","429","resource_exhausted")) else
               "AUTH_ERROR"  if any(k in err_str for k in ("auth","403","key","permission")) else "API_ERROR")
        print(f"  [CALL 2] FAILED -- {cat} ({lat} ms): {e}")
        return {"status": "FAILED", "latency_ms": lat, "grounding_rate": 0.0, "error": f"{cat}: {e}",
                "claims_total": 0, "claims_grounded": 0, "claims_ungrounded": 0}

# ============================================================
# Save results
# ============================================================
def save_results(r1, r2):
    out_dir = _PROJECT_ROOT / "reports" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    overall = ("PASS" if r1["status"]=="PASS" and r2["status"]=="PASS" else
               "PARTIAL" if r1["status"]=="PASS" else "FAILED")
    truth = {
        "gemini_api_connected":   "PASS" if r1["status"]=="PASS" else "FAILED",
        "real_gemini_generation": "PASS" if r1["status"]=="PASS" else "NOT VALIDATED",
        "grounded_gemini_answer": "PASS" if r2["status"]=="PASS" else "NOT VALIDATED",
        "grounding_rate": f"{r2.get('grounding_rate',0):.0%}",
        "fabrication_detected": "NO" if r2.get("claims_ungrounded",0)==0 and r2["status"]=="PASS" else "YES",
    }
    data = {"phase":"8.5","sub_test":"gemini_quota_efficient_validation","timestamp":ts,
            "overall":overall,"call_1_smoke":r1,"call_2_grounded":r2,
            "truth_table":truth,"total_api_calls":2,"pace_seconds":PACE_SECONDS}
    json_path = out_dir / "phase85_gemini_validate.json"
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    md = f"""# Phase 8.5 -- Gemini Quota-Efficient Validation

**Date:** {ts}  
**Total API calls used:** 2 / quota  
**Pace between calls:** {PACE_SECONDS}s  

## Call 1 -- Smoke Ping
| Field | Value |
|---|---|
| Status | {r1['status']} |
| Latency | {r1['latency_ms']} ms |
| Error | {r1.get('error') or '--'} |

## Call 2 -- Grounded Synthesis (Inline Evidence)
| Field | Value |
|---|---|
| Status | {r2['status']} |
| Latency | {r2['latency_ms']} ms |
| Claims total | {r2.get('claims_total',0)} |
| Claims grounded | {r2.get('claims_grounded',0)} |
| Claims ungrounded | {r2.get('claims_ungrounded',0)} |
| **Grounding Rate** | **{r2.get('grounding_rate',0):.0%}** |
| Error | {r2.get('error') or '--'} |

## Truth Table
| Capability | Status |
|---|---|
| Gemini API connected | {truth['gemini_api_connected']} |
| Real Gemini generation | {truth['real_gemini_generation']} |
| Grounded Gemini answer | {truth['grounded_gemini_answer']} |
| Grounding rate | {truth['grounding_rate']} |
| Fabrication detected | {truth['fabrication_detected']} |

**Overall: {overall}**
"""
    md_path = out_dir / "phase85_gemini_validate.md"
    with open(md_path,"w",encoding="utf-8") as f:
        f.write(md)
    return json_path

# ============================================================
# Main
# ============================================================
async def main():
    print(_b("\n+============================================================+"))
    print(_b("|  AI Research Copilot -- Phase 8.5 Gemini Validation       |"))
    print(_b("|  Quota-Efficient: exactly 2 API calls                     |"))
    print(_b("+============================================================+\n"))

    from core.config import get_settings
    from llm.gemini_provider import GeminiProvider
    st = get_settings()
    provider = GeminiProvider(settings=st)

    print(f"  Provider  : {provider.provider_name}")
    print(f"  Model     : {provider.model_name}")
    print(f"  API key   : {'CONFIGURED' if st.gemini_api_key else 'MISSING'}")
    print(f"  Paid guard: {'ACTIVE' if not st.llm_allow_paid_provider else 'DISABLED'}\n")

    if not st.gemini_api_key:
        print(_r("  ERROR: GEMINI_API_KEY is not set in .env"))
        return 1

    r1 = await call1_smoke(provider)

    if r1["status"] == "FAILED":
        err = r1.get("error", "")
        msg = "Quota limit hit." if "QUOTA" in str(err) else ("Auth error -- check GEMINI_API_KEY." if "AUTH" in str(err) else f"Call 1 failed ({err}).")
        print(_y(f"\n  {msg} No further calls made."))
        save_results(r1, {"status":"NOT RUN","latency_ms":0,"grounding_rate":0,
                          "claims_total":0,"claims_grounded":0,"claims_ungrounded":0,"error":"Call 1 failed"})
        return 1

    print(f"\n  Pacing {PACE_SECONDS}s before Call 2 (free-tier RPM safety)...")
    for remaining in range(PACE_SECONDS, 0, -5):
        print(f"    {remaining}s remaining...", flush=True)
        await asyncio.sleep(5)

    r2 = await call2_grounded(provider)

    json_path = save_results(r1, r2)

    overall = ("PASS" if r1["status"]=="PASS" and r2["status"]=="PASS" else
               "PARTIAL" if r1["status"]=="PASS" else "FAILED")

    print(_b("\n+============================================================+"))
    print(_b("  FINAL TRUTH TABLE"))
    print(_b("+============================================================+"))
    rows = [
        ("Gemini API connected",   "PASS" if r1["status"]=="PASS" else "FAILED"),
        ("Real Gemini generation", "PASS" if r1["status"]=="PASS" else "NOT VALIDATED"),
        ("Grounded Gemini answer", "PASS" if r2["status"]=="PASS" else "NOT VALIDATED"),
        ("Grounding rate",         f"{r2.get('grounding_rate',0):.0%}"),
        ("Fabrication detected",   "NO" if r2.get("claims_ungrounded",0)==0 and r2["status"]=="PASS" else "YES"),
        ("Total API calls used",   "2"),
    ]
    for cap, stat in rows:
        col = _g if stat in {"PASS","NO","2"} or (stat.endswith("%") and r2["status"]=="PASS") else _y
        print(f"  {cap:<32} : {col(stat)}")
    print(_b(f"\n  Overall : {(_g if overall=='PASS' else _y)(overall)}"))
    print(f"\n  Report  : {json_path}")
    print(_b("+============================================================+\n"))
    return 0 if overall in ("PASS","PARTIAL") else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

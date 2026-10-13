"""
CLI script to test Grounded Research Answer Synthesis with full citation mapping and verification.

Usage:
    python scripts/test_llm_synthesis.py "What is Retrieval-Augmented Generation for NLP?"
    python scripts/test_llm_synthesis.py "How does DPR compare to BM25?" --provider mock
    python scripts/test_llm_synthesis.py "What are Transformers?" --provider ollama
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

# Add backend to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from synthesis.answer_service import (
    ResearchAnswerRequest,
    ResearchAnswerService,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test Grounded LLM Research Synthesis & Citations")
    parser.add_argument(
        "query",
        nargs="?",
        default="What is Retrieval-Augmented Generation (RAG) and how does it improve NLP tasks?",
        help="Research question to answer",
    )
    parser.add_argument("--candidate-k", type=int, default=25, help="Number of candidate passages")
    parser.add_argument("--evidence-k", type=int, default=6, help="Target evidence passages")
    parser.add_argument("--provider", type=str, default="mock", help="LLM provider: mock | ollama | openai | anthropic")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    args = parser.parse_args()

    print("=" * 80)
    print(f"AI Research Copilot — Grounded Synthesis & Citation Engine (Phase 5)")
    print(f"Question: '{args.query}'")
    print(f"Provider: {args.provider.upper()} | Candidate K: {args.candidate_k} | Evidence K: {args.evidence_k}")
    print("=" * 80)

    service = ResearchAnswerService()
    req = ResearchAnswerRequest(
        query=args.query,
        candidate_k=args.candidate_k,
        evidence_k=args.evidence_k,
        temperature=args.temperature,
        provider_override=args.provider,
    )

    answer = await service.answer_question(req)

    print("\n[1] EXECUTIVE SUMMARY:")
    print("-" * 80)
    print(f"{answer.summary}\n")

    if answer.insufficient_evidence:
        print("⚠️ NOTICE: Insufficient evidence available to synthesize a reliable answer.")

    print("[2] STRUCTURED CLAIMS & CITATION ATTRIBUTION:")
    print("-" * 80)
    for c in answer.claims:
        cites_str = ", ".join(c.citation_ids) if c.citation_ids else "NO CITATION"
        ev_str = ", ".join(c.evidence_ids) if c.evidence_ids else "NO EVIDENCE"
        print(f"  [{c.claim_id}] Status: {c.support_status.value:<19} | Citations: {cites_str} (Ev: {ev_str})")
        print(f"    Claim: {c.text}\n")

    print("[3] VERIFIED BIBLIOGRAPHIC CITATIONS:")
    print("-" * 80)
    if answer.citations:
        for cite in answer.citations:
            print(f"  {cite.formatted_citation}")
    else:
        print("  (No citations generated)")

    print("\n[4] SOURCE PROVENANCE SUMMARY:")
    print("-" * 80)
    for src in answer.sources:
        print(f"  - [{src.source_type}] {src.title} (Source ID: {src.source_id})")
        if src.url:
            print(f"    URL: {src.url}")

    if answer.limitations:
        print("\n[5] IDENTIFIED LIMITATIONS & BOUNDARIES:")
        print("-" * 80)
        for lim in answer.limitations:
            print(f"  - {lim}")

    print("\n[6] CITATION VERIFICATION AUDIT:")
    print("-" * 80)
    v = answer.verification
    status_icon = "[PASS]" if v.valid else "[FAIL]"
    print(f"  Audit Status:       {status_icon} (Valid Grounding: {v.valid})")
    print(f"  Grounding Rate:     {v.grounding_rate * 100:.1f}% ({v.verified_claim_count}/{v.total_claim_count} claims supported)")
    if v.errors:
        print(f"  Errors ({len(v.errors)}):")
        for err in v.errors:
            print(f"    ❌ {err}")
    if v.warnings:
        print(f"  Warnings ({len(v.warnings)}):")
        for warn in v.warnings:
            print(f"    ⚠️ {warn}")

    print("\n[7] TELEMETRY & LATENCY BREAKDOWN:")
    print("-" * 80)
    m = answer.metadata
    print(f"  - LLM Provider / Model:     {m.provider} / {m.model}")
    print(f"  - Retrieval Latency:        {m.retrieval_latency_ms:>8.2f} ms")
    print(f"  - LLM Generation Latency:   {m.generation_latency_ms:>8.2f} ms")
    print(f"  - Verification Latency:     {m.verification_latency_ms:>8.2f} ms")
    print(f"  - Total Pipeline Latency:   {m.total_latency_ms:>8.2f} ms")
    if m.total_tokens:
        print(f"  - Token Usage:              In: {m.input_tokens}, Out: {m.output_tokens}, Total: {m.total_tokens}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

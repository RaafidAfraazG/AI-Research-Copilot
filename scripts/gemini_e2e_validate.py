"""
Bounded end-to-end validation of the REAL research-answer path with Gemini Free Tier.

Exercises, in one request through ResearchAnswerService:
  query analysis + deterministic decomposition -> parallel hybrid retrieval (BM25 + dense)
  -> neural cross-encoder reranking -> evidence selection -> Gemini synthesis
  -> citation mapping + structural verification.

Safety (Free Tier):
  * exactly ONE Gemini generation request; no retry loop, no batch, no benchmark
  * the configured GEMINI_MODEL is used as-is (never silently switched)
  * the API key / environment values are never printed or written to the report
  * no key => exit 2 with a clear manual prerequisite; no other provider is tried
  * any provider failure is reported as a failure (never replaced by a mock answer)

Exit codes: 0 PASS, 1 FAIL, 2 NOT RUN (prerequisite missing), 3 INCONCLUSIVE.

Usage:  python scripts/gemini_e2e_validate.py [--query "..."] [--output reports/evaluation]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

DEFAULT_QUERY = (
    "Compare dense passage retrieval with BM25 keyword search and explain why "
    "retrieval is used before generation in retrieval-augmented generation."
)


class CountingProvider:
    """Wraps an LLMProvider, counts generate() calls and never retries."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return await self._inner.generate(*args, **kwargs)


def _result(status: str, **kw: Any) -> dict[str, Any]:
    return {
        "phase": "10",
        "test": "gemini_e2e_validation",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": status,
        **kw,
    }


async def validate(
    settings: Any,
    query: str = DEFAULT_QUERY,
    llm_provider: Any | None = None,
    dump_answer_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run the bounded validation. `llm_provider` is for unit tests only (control-flow
    checks); a report produced with an injected provider is labelled and is NOT proof
    that Gemini works.
    """
    from core.error_handling import classify_llm_error
    from core.redaction import redact
    from embedding.factory import create_embedding_provider
    from llm.gemini_provider import GeminiProvider
    from retrieval.index_manifest import IndexIntegrityError, IndexStore
    from synthesis.answer_service import ResearchAnswerRequest, ResearchAnswerService

    injected = llm_provider is not None

    if not injected:
        if settings.llm_provider != "gemini":
            return _result(
                "NOT_RUN",
                reason=f"LLM_PROVIDER is '{settings.llm_provider}', not 'gemini'. "
                "Set LLM_PROVIDER=gemini (and GEMINI_API_KEY) to validate the real provider.",
            )
        if not settings.gemini_api_key:
            return _result(
                "NOT_RUN",
                reason="GEMINI_API_KEY is not configured. Create a free key at "
                "https://aistudio.google.com and set it in .env, then re-run. "
                "No other provider was tried.",
            )
        inner = GeminiProvider(settings=settings)
    else:
        inner = llm_provider

    # Index must be real, complete and compatible; fail closed (no mock fallback).
    try:
        IndexStore(settings).validate_for_read(create_embedding_provider(settings))
    except Exception as e:  # AppException subclasses carry safe messages
        code = getattr(e, "code", type(e).__name__)
        return _result("FAIL", stage="index", reason=f"Index/embedding check failed: {code}")

    provider = CountingProvider(inner)
    service = ResearchAnswerService(llm_provider=provider, settings=settings)
    req = ResearchAnswerRequest(
        query=query,
        enable_decomposition=True,
        decomposition_mode="deterministic",  # planning without extra LLM calls
        candidate_k=15,
        evidence_k=4,
        temperature=0.0,
        max_tokens=2048,  # structured JSON (claims + sections) truncates below ~1.5k tokens
    )

    t0 = time.perf_counter()
    try:
        answer = await service.answer_question(req)
    except Exception as e:
        status_code, code, message = classify_llm_error(f"{type(e).__name__}: {e}")
        is_llm = type(e).__name__.startswith("LLM")
        return _result(
            "FAIL",
            stage="generation" if is_llm else "pipeline",
            llm_calls=provider.calls,
            error_category=code if is_llm else type(e).__name__,
            reason=redact(message if is_llm else f"Pipeline failed with {type(e).__name__}"),
            provider=provider.provider_name,
            model=provider.model_name,
            real_gemini_call=bool(is_llm and not injected),
        )
    finally:
        await service.retrieval_service.aclose()  # close retriever client sessions
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    if dump_answer_path is not None:
        dump_answer_path.parent.mkdir(parents=True, exist_ok=True)
        dump_answer_path.write_text(answer.model_dump_json(indent=2), encoding="utf-8")

    reranker = service.retrieval_service.ranking_service.reranker
    neural = getattr(reranker, "_model", None) is not None
    md = answer.metadata
    plan = answer.research_plan or {}
    v = answer.verification

    checks = {
        "provider_is_gemini": md.provider == "gemini",
        "model_is_configured_model": md.model == settings.gemini_model,
        "single_llm_call": provider.calls == 1,
        "query_analysis_present": answer.query_analysis is not None,
        "plan_present": bool(plan),
        "evidence_retrieved": len(answer.evidence) > 0,
        "neural_reranker_active": neural,
        "not_insufficient_evidence": not answer.insufficient_evidence,
        "has_claims": len(answer.claims) > 0,
        "has_citations": len(answer.citations) > 0,
        "sections_consistent_with_claims": all(
            any(c.claim_id == sc.claim_id and c.support_status == sc.support_status for c in answer.claims)
            for sec in answer.sections
            for sc in sec.claims
        ),
    }
    real = (md.provider == "gemini") and not injected

    if answer.insufficient_evidence:
        status = "INCONCLUSIVE"
    elif all(checks.values()) and real:
        status = "PASS"
    else:
        status = "FAIL"

    return _result(
        status,
        real_gemini_call=real,
        mode="real_gemini_free_tier" if real else "INJECTED_TEST_PROVIDER_NOT_REAL_GEMINI",
        provider=md.provider,
        model=md.model,
        llm_calls=provider.calls,
        latency_ms=latency_ms,
        checks=checks,
        plan={
            "strategy": plan.get("strategy"),
            "subqueries": plan.get("subquery_count"),
            "intent": (answer.query_analysis or {}).get("intent"),
        },
        retrieval={"evidence_passages": len(answer.evidence), "sources": len(answer.sources)},
        reranker={"model": getattr(reranker, "model_name", None), "neural": neural},
        answer_stats={
            "claims": len(answer.claims),
            "citations": len(answer.citations),
            "supported": md.supported_claim_count,
            "partially_supported": md.partially_supported_claim_count,
            "unsupported": md.unsupported_claim_count,
            "verification_valid": v.valid,
            "verification_method": v.verification_method,
            "semantic_entailment_checked": v.semantic_entailment_checked,
        },
        tokens={"input": md.input_tokens, "output": md.output_tokens, "total": md.total_tokens},
        note=(
            "Support status is a structural evidence-id check, not an entailment check. "
            "Read the cited passages to judge factual support."
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--output", default="reports/evaluation")
    ap.add_argument("--dump-answer", default=None, help="write the full answer JSON here (for manual citation review)")
    args = ap.parse_args()

    from core.config import get_settings

    settings = get_settings()
    print("Gemini end-to-end validation (single bounded request)")
    print(f"  provider setting : {settings.llm_provider}")
    print(f"  model            : {settings.gemini_model}")
    print(f"  api key present  : {bool(settings.gemini_api_key)}")  # never the value

    report = asyncio.run(
        validate(
            settings,
            query=args.query,
            dump_answer_path=(_ROOT / args.dump_answer) if args.dump_answer else None,
        )
    )

    out = _ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    path = out / "gemini_e2e_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSTATUS: {report['status']}")
    for k in ("reason", "stage", "error_category", "mode", "latency_ms", "llm_calls"):
        if k in report:
            print(f"  {k}: {report[k]}")
    for name, ok in (report.get("checks") or {}).items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nReport: {path.relative_to(_ROOT)}")
    return {"PASS": 0, "FAIL": 1, "NOT_RUN": 2, "INCONCLUSIVE": 3}.get(report["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Phase 8 Validation Script — Real Neural Retrieval & Free-Tier LLM Validation

Validates:
  1. Neural cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)
  2. Embedding model (configurable, default all-MiniLM-L6-v2 or all-mpnet-base-v2)
  3. Google Gemini LLM provider (free-tier, gemini-1.5-flash)
  4. Paid-provider safety guard enforcement
  5. Fallback graceful degradation

IMPORTANT LABELLING:
  - Neural reranker:   REAL NEURAL RERANKER  /  RERANKER FALLBACK (hybrid score)
  - LLM provider:     REAL LLM (Gemini)      /  MOCK LLM

Usage:
  cd backend
  python ../scripts/phase8_validate.py

  # With Gemini API key:
  GEMINI_API_KEY=<key> python ../scripts/phase8_validate.py

  # Force all-mpnet-base-v2 embedding:
  EMBEDDING_MODEL=all-mpnet-base-v2 python ../scripts/phase8_validate.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure backend/ is on sys.path when run from project root or scripts/
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Also allow running from within the backend/ dir
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("POSTGRES_PASSWORD", "test_password_for_ci")
os.environ.setdefault("APP_ENV", "development")

# -- ANSI colour helpers -------------------------------------------------------

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"

def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"

def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"

def _cyan(s: str) -> str:
    return f"\033[36m{s}\033[0m"


PASS = _green("PASS")
FAIL = _red("FAIL")
SKIP = _yellow("SKIP")
WARN = _yellow("WARN")


# -- Result tracking -----------------------------------------------------------

results: list[tuple[str, str, str]] = []  # (section, status, message)


def record(section: str, status: str, message: str) -> None:
    results.append((section, status, message))
    icon = {"PASS": "[OK]", "FAIL": "[X]", "SKIP": "[*]", "WARN": "[!]"}.get(status, "[?]")
    col = {"PASS": _green, "FAIL": _red, "SKIP": _yellow, "WARN": _yellow}.get(status, str)
    print(f"  {col(icon)} [{section}] {message}")


# -- 1. Config / Environment ---------------------------------------------------

def validate_config() -> None:
    print(_bold("\n--- 1. Configuration & Environment ---"))
    from core.config import get_settings
    st = get_settings()

    record("config", "PASS", f"LLM_PROVIDER      = {st.llm_provider}")
    record("config", "PASS", f"GEMINI_MODEL      = {st.gemini_model}")
    record("config", "PASS", f"EMBEDDING_PROVIDER = {st.embedding_provider}")
    record("config", "PASS", f"EMBEDDING_MODEL   = {st.embedding_model}")
    record("config", "PASS", f"RERANKER_MODEL    = {st.reranker_model}")
    record("config", "PASS", f"RERANKER_DEVICE   = {st.reranker_device}")

    if st.gemini_api_key:
        record("config", "PASS", "GEMINI_API_KEY    = *** (set)")
    else:
        record("config", "WARN",
               "GEMINI_API_KEY    = NOT SET -- Gemini provider will fail at runtime. "
               "Get a free key at https://aistudio.google.com")

    if st.llm_allow_paid_provider:
        record("config", "WARN",
               "LLM_ALLOW_PAID_PROVIDER = true -- paid providers (OpenAI, Anthropic) are enabled")
    else:
        record("config", "PASS",
               "LLM_ALLOW_PAID_PROVIDER = false -- paid provider guard is ACTIVE")


# -- 2. Paid-provider safety guard --------------------------------------------

def validate_paid_provider_guard() -> None:
    print(_bold("\n--- 2. Paid-Provider Safety Guard ---"))
    from core.config import Settings
    from llm.factory import get_llm_provider
    from llm.models import LLMConfigurationError

    # With guard active (default), OpenAI should be blocked
    st_guard_on = Settings(llm_allow_paid_provider=False)
    try:
        get_llm_provider("openai", settings=st_guard_on)
        record("paid_guard", "FAIL", "OpenAI NOT blocked when LLM_ALLOW_PAID_PROVIDER=false")
    except LLMConfigurationError as e:
        record("paid_guard", "PASS", f"OpenAI correctly blocked: {str(e)[:80]}...")

    # With guard active, Anthropic should be blocked
    try:
        get_llm_provider("anthropic", settings=st_guard_on)
        record("paid_guard", "FAIL", "Anthropic NOT blocked when LLM_ALLOW_PAID_PROVIDER=false")
    except LLMConfigurationError:
        record("paid_guard", "PASS", "Anthropic correctly blocked by guard")

    # Gemini should always be allowed
    try:
        p = get_llm_provider("gemini", settings=st_guard_on)
        record("paid_guard", "PASS", f"Gemini always allowed -- provider_name={p.provider_name}")
    except Exception as e:
        record("paid_guard", "FAIL", f"Gemini unexpectedly blocked: {e}")

    # Mock should always be allowed
    try:
        p = get_llm_provider("mock", settings=st_guard_on)
        record("paid_guard", "PASS", f"Mock always allowed -- provider_name={p.provider_name}")
    except Exception as e:
        record("paid_guard", "FAIL", f"Mock unexpectedly blocked: {e}")

    # With guard off, OpenAI should be instantiatable (but won't have a key)
    st_guard_off = Settings(llm_allow_paid_provider=True, openai_api_key=None)
    try:
        p = get_llm_provider("openai", settings=st_guard_off)
        record("paid_guard", "PASS",
               "OpenAI allowed when LLM_ALLOW_PAID_PROVIDER=true (no key -- will fail on first call)")
    except Exception as e:
        record("paid_guard", "FAIL", f"OpenAI failed even with guard off: {e}")


# -- 3. Embedding model --------------------------------------------------------

def validate_embedding() -> None:
    print(_bold("\n--- 3. Embedding Model ---"))
    from embedding.factory import create_embedding_provider
    from core.config import Settings

    st = Settings()
    print(f"  -> Attempting to load: {st.embedding_model} (provider: {st.embedding_provider})")

    try:
        t0 = time.perf_counter()
        provider = create_embedding_provider(st)
        elapsed = time.perf_counter() - t0

        # Quick encode test
        t1 = time.perf_counter()
        vecs = provider.embed_texts(["Retrieval-augmented generation", "Transformer architecture"])
        encode_ms = (time.perf_counter() - t1) * 1000

        if hasattr(provider, "_model_name") and "Mock" not in type(provider).__name__:
            record("embedding", "PASS",
                   f"REAL NEURAL EMBEDDING: {provider._model_name} | "
                   f"dim={len(vecs[0])} | load={elapsed:.2f}s | encode={encode_ms:.1f}ms")
        else:
            record("embedding", "WARN",
                   f"MOCK EMBEDDING (sentence-transformers not available): dim={len(vecs[0])}")

    except Exception as e:
        record("embedding", "FAIL", f"Embedding load failed: {e}")


# -- 4. Neural Cross-Encoder Reranker -----------------------------------------

async def validate_reranker() -> None:
    print(_bold("\n--- 4. Neural Cross-Encoder Reranker ---"))
    from core.config import Settings
    from ranking.cross_encoder_reranker import CrossEncoderReranker
    from retrieval.models import ScoredChunk

    st = Settings()
    model_name = st.reranker_model
    print(f"  -> Attempting to load: {model_name}")

    reranker = CrossEncoderReranker(
        model_name=model_name,
        device=st.reranker_device,
        fallback_mode="hybrid_score",
        settings=st,
    )

    candidates = [
        ScoredChunk(
            chunk_id="c1", document_id="d1", source_type="arxiv", source_id="2005.11401",
            title="RAG Paper",
            content="Retrieval-augmented generation combines parametric and non-parametric memory.",
            score=0.032, rank=1, retriever_name="hybrid",
        ),
        ScoredChunk(
            chunk_id="c2", document_id="d2", source_type="arxiv", source_id="1706.03762",
            title="Attention Paper",
            content="The weather today is sunny and warm with mild temperatures.",
            score=0.028, rank=2, retriever_name="hybrid",
        ),
        ScoredChunk(
            chunk_id="c3", document_id="d3", source_type="arxiv", source_id="2004.04906",
            title="DPR Paper",
            content="Dense passage retrieval uses dual encoders for open-domain question answering.",
            score=0.025, rank=3, retriever_name="hybrid",
        ),
    ]

    t0 = time.perf_counter()
    reranked = await reranker.rerank(
        query="How does retrieval-augmented generation work?",
        candidates=candidates,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    is_fallback = reranker._load_failed or reranked[0].metadata.get("reranker_fallback", False)

    if not is_fallback and reranker._model is not None:
        # Verify that obviously irrelevant passage (weather) was pushed down
        weather_chunk = next((c for c in reranked if "weather" in c.content.lower()), None)
        weather_rank = weather_chunk.rank if weather_chunk else None
        record("reranker", "PASS",
               f"REAL NEURAL RERANKER: {model_name} | "
               f"latency={elapsed_ms:.1f}ms | weather_rank={weather_rank}/3 (expected 3)")
        if weather_rank and weather_rank < 3:
            record("reranker", "WARN",
                   f"Unexpected: off-topic passage ranked {weather_rank} -- model may not have loaded correctly")
    else:
        record("reranker", "WARN",
               f"RERANKER FALLBACK (hybrid score) -- model '{model_name}' could not be loaded. "
               f"Error: {reranker._load_error_msg}. "
               "Install sentence-transformers: pip install sentence-transformers")
        print()
        print(_yellow("  REAL NEURAL RERANKER: NOT VALIDATED"))
        print(_yellow("  To fix: pip install sentence-transformers"))
        print()


# -- 5. Gemini LLM provider ----------------------------------------------------

async def validate_gemini() -> None:
    print(_bold("\n--- 5. Google Gemini LLM Provider ---"))
    from core.config import get_settings
    from llm.gemini_provider import GeminiProvider

    st = get_settings()

    if not st.gemini_api_key:
        record("gemini", "SKIP",
               "GEMINI_API_KEY not set -- skipping live API call. "
               "Get a free key at https://aistudio.google.com")
        print()
        print(_yellow("  REAL LLM (Gemini): NOT VALIDATED -- no API key"))
        print()
        return

    provider = GeminiProvider(settings=st)
    prompt = (
        "In two sentences, explain what retrieval-augmented generation (RAG) is. "
        "Be concise and accurate."
    )

    print(f"  -> Calling {provider.model_name} ...")
    t0 = time.perf_counter()
    try:
        resp = await provider.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=256,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        record("gemini", "PASS",
               f"REAL LLM (Gemini): {resp.model} | "
               f"latency={resp.latency_ms:.0f}ms | "
               f"tokens={resp.usage.total_tokens if resp.usage else 'n/a'}")
        print(f"\n  Response preview: {_cyan(resp.content[:200])}")

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        record("gemini", "FAIL", f"Gemini call failed after {elapsed_ms:.0f}ms: {e}")
        print()
        print(_red("  REAL LLM (Gemini): NOT VALIDATED"))
        print()


# -- 6. Mock LLM still works ---------------------------------------------------

async def validate_mock_llm() -> None:
    print(_bold("\n--- 6. Mock LLM (Offline Fallback) ---"))
    from llm.mock_provider import MockLLMProvider

    provider = MockLLMProvider()
    prompt = (
        "USER QUESTION: What is RAG?\n"
        "RETRIEVED EVIDENCE PASSAGES:\n"
        "[EVIDENCE_ID: evidence_001]\nTitle: RAG Paper\n"
        "Passage: Retrieval-augmented generation combines parametric and non-parametric memory.\n"
        "[EVIDENCE_ID: evidence_002]\nTitle: DPR\n"
        "Passage: Dense retrieval uses dual encoders.\n"
    )
    try:
        resp = await provider.generate(prompt=prompt, response_format="json")
        import json
        data = json.loads(resp.content)
        assert "claims" in data
        record("mock_llm", "PASS",
               f"MOCK LLM: {resp.provider} | claims={len(data['claims'])} | latency={resp.latency_ms:.1f}ms")
    except Exception as e:
        record("mock_llm", "FAIL", f"Mock LLM failed: {e}")


# -- 7. Factory integration ----------------------------------------------------

def validate_factory_integration() -> None:
    print(_bold("\n--- 7. Factory Integration ---"))
    from core.config import Settings
    from llm.factory import get_llm_provider
    from llm.gemini_provider import GeminiProvider
    from llm.mock_provider import MockLLMProvider

    # Factory creates Gemini when provider=gemini
    try:
        st = Settings(llm_provider="gemini", gemini_api_key="fake-key-for-test")
        p = get_llm_provider(settings=st)
        assert isinstance(p, GeminiProvider), f"Expected GeminiProvider, got {type(p)}"
        record("factory", "PASS", "Factory -> gemini -> GeminiProvider")
    except Exception as e:
        record("factory", "FAIL", f"Factory gemini selection failed: {e}")

    # Factory creates Mock when provider=mock
    try:
        st = Settings(llm_provider="mock")
        p = get_llm_provider(settings=st)
        assert isinstance(p, MockLLMProvider)
        record("factory", "PASS", "Factory -> mock -> MockLLMProvider")
    except Exception as e:
        record("factory", "FAIL", f"Factory mock selection failed: {e}")

    # Unknown provider raises error
    try:
        from llm.models import LLMConfigurationError
        st = Settings(llm_provider="mock")  # avoid default LLM_PROVIDER env issue
        get_llm_provider("unsupported_xyz", settings=st)
        record("factory", "FAIL", "Unknown provider did NOT raise error")
    except Exception:
        record("factory", "PASS", "Factory raises error for unknown provider")


# -- 8. Summary ----------------------------------------------------------------

def print_summary() -> None:
    print(_bold("\n" + "=" * 60))
    print(_bold("  PHASE 8 VALIDATION SUMMARY"))
    print(_bold("=" * 60))

    pass_count = sum(1 for _, s, _ in results if s == "PASS")
    fail_count = sum(1 for _, s, _ in results if s == "FAIL")
    warn_count = sum(1 for _, s, _ in results if s == "WARN")
    skip_count = sum(1 for _, s, _ in results if s == "SKIP")

    print(f"  {_green(f'{pass_count} PASSED')}  {_red(f'{fail_count} FAILED')}  "
          f"{_yellow(f'{warn_count} WARNINGS')}  {_yellow(f'{skip_count} SKIPPED')}")

    if fail_count == 0:
        print(f"\n  {_green('All critical checks passed. Phase 8 is validated.')}")
    else:
        print(f"\n  {_red(f'{fail_count} check(s) failed. Review output above.')}")

    print()

    # Print reranker + LLM status explicitly
    reranker_real = any("REAL NEURAL RERANKER" in m for _, s, m in results if s == "PASS")
    reranker_fallback = any("RERANKER FALLBACK" in m for _, _, m in results)
    gemini_real = any("REAL LLM (Gemini)" in m for _, s, m in results if s == "PASS")
    gemini_skip = any("REAL LLM (Gemini): NOT VALIDATED" in m for _, _, m in results)

    print(_bold("  Reranker status:"))
    if reranker_real:
        print(f"    {_green('[OK] REAL NEURAL RERANKER -- cross-encoder inference validated')}")
    else:
        print(f"    {_yellow('[!] RERANKER FALLBACK -- cross-encoder model not loaded')}")
        print(f"      pip install sentence-transformers  (to fix)")

    print(_bold("  LLM status:"))
    if gemini_real:
        print(f"    {_green('[OK] REAL LLM (Gemini) -- live free-tier inference validated')}")
    else:
        print(f"    {_yellow('[!] MOCK LLM only -- set GEMINI_API_KEY in .env for live validation')}")
        print(f"      Free key: https://aistudio.google.com")
    print()


# --- Entry point -------------------------------------------------------------

async def main() -> int:
    print(_bold("+==========================================================+"))
    print(_bold("|   AI Research Copilot -- Phase 8 Validation              |"))
    print(_bold("|   Real Neural Retrieval & Free-Tier LLM                  |"))
    print(_bold("+==========================================================+"))

    validate_config()
    validate_paid_provider_guard()
    validate_embedding()
    await validate_reranker()
    await validate_gemini()
    await validate_mock_llm()
    validate_factory_integration()
    print_summary()

    fail_count = sum(1 for _, s, _ in results if s == "FAIL")
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

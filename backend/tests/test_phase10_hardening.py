"""
Phase 10 focused tests: SSRF/redirect/streaming limits, reranker reuse, error
sanitization, client lifecycle, readiness semantics, claim/verification consistency
and the Gemini validation script's control flow.

Everything is offline: network behavior is mocked (httpx.MockTransport + injected
resolver), providers are fakes. NOTHING here proves real Gemini works.
"""
from __future__ import annotations

import asyncio
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from connectors.http_client import (
    ResponseTooLargeError,
    SafeAsyncHttpClient,
    TooManyRedirectsError,
)
from connectors.url_security import SSRFValidationError, check_ip

# Fake Google-API-key-shaped value used to prove redaction. It is assembled from fragments so no
# key-shaped literal appears in the source (secret scanners flag the literal shape).
_FAKE_GOOGLE_KEY = "AI" + "za" + "Sy" + "FAKE" * 8 + "1"

# ─────────────────────────── SSRF / outbound HTTP ────────────────────────────

PUBLIC = {"example.com": ["93.184.216.34"], "other.example.org": ["8.8.8.8"]}


def _resolver(mapping: dict[str, list[str]]):
    def resolve(host: str) -> list[str]:
        if host in mapping:
            return mapping[host]
        return [host]  # IP literals resolve to themselves

    return resolve


def _client(handler, resolver=None, **kw) -> SafeAsyncHttpClient:
    return SafeAsyncHttpClient(
        timeout_seconds=5,
        max_retries=0,
        min_request_interval_seconds=0.0,
        transport=httpx.MockTransport(handler),
        resolver=resolver or _resolver(PUBLIC),
        **kw,
    )


@pytest.mark.parametrize(
    "addr",
    [
        "127.0.0.1", "10.1.2.3", "192.168.0.9", "172.16.5.5", "169.254.169.254",
        "0.0.0.0", "100.64.0.1", "224.0.0.1", "::1", "fe80::1", "fc00::1",
        "::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:192.168.1.1",
    ],
)
def test_check_ip_rejects_non_public(addr):
    with pytest.raises(SSRFValidationError):
        check_ip(addr)


def test_check_ip_allows_public():
    check_ip("93.184.216.34")
    check_ip("2606:2800:220:1:248:1893:25c8:1946")


@pytest.mark.asyncio
async def test_redirect_to_private_address_is_blocked():
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        return httpx.Response(302, headers={"Location": "http://10.0.0.5/admin"})

    client = _client(handler)
    with pytest.raises(SSRFValidationError):
        await client.get("https://example.com/start", ssrf_protect=True, max_bytes=1000)
    assert len(seen) == 1  # the private hop was never requested


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    [
        "http://127.0.0.1:8000/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost/",
    ],
)
async def test_redirect_to_loopback_and_mapped_ipv6_blocked(location):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(301, headers={"Location": location})

    with pytest.raises(SSRFValidationError):
        await _client(handler).get("https://example.com/", ssrf_protect=True)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_alternate_numeric_ip_forms_blocked_via_resolution():
    """Decimal/hex/octal spellings resolve to loopback; validation is on the resolved IP."""
    resolver = _resolver({"2130706433": ["127.0.0.1"], "0x7f.1": ["127.0.0.1"], **PUBLIC})
    hits = []

    def handler(req):
        hits.append(req)
        return httpx.Response(302, headers={"Location": "http://2130706433/"})

    with pytest.raises(SSRFValidationError):
        await _client(handler, resolver).get("https://example.com/", ssrf_protect=True)
    with pytest.raises(SSRFValidationError):
        await _client(lambda r: httpx.Response(200), resolver).get("http://0x7f.1/", ssrf_protect=True)
    assert len(hits) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com/"])
async def test_unsupported_initial_schemes_rejected(url):
    called = []
    client = _client(lambda r: called.append(r) or httpx.Response(200))
    with pytest.raises(SSRFValidationError, match="scheme"):
        await client.get(url, ssrf_protect=True)
    assert not called


@pytest.mark.asyncio
async def test_redirect_to_unsupported_scheme_rejected():
    def handler(req):
        return httpx.Response(302, headers={"Location": "file:///etc/passwd"})

    with pytest.raises(SSRFValidationError, match="scheme"):
        await _client(handler).get("https://example.com/", ssrf_protect=True)


@pytest.mark.asyncio
async def test_credentials_in_url_rejected():
    with pytest.raises(SSRFValidationError, match="credentials"):
        await _client(lambda r: httpx.Response(200)).get("https://user:pw@example.com/", ssrf_protect=True)


@pytest.mark.asyncio
async def test_safe_public_redirect_still_works_and_is_pinned():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.host, req.headers.get("host")))
        if req.url.host == "93.184.216.34":
            return httpx.Response(302, headers={"Location": "https://other.example.org/final"})
        return httpx.Response(200, text="hello", headers={"content-type": "text/plain"})

    res = await _client(handler).get("https://example.com/a", ssrf_protect=True, max_bytes=1000)
    assert res.status_code == 200 and res.text == "hello"
    # connected to the validated IP literals, with the original Host header
    assert seen == [("93.184.216.34", "example.com"), ("8.8.8.8", "other.example.org")]


@pytest.mark.asyncio
async def test_redirect_loop_is_bounded():
    def handler(req):
        return httpx.Response(302, headers={"Location": "https://example.com/again"})

    with pytest.raises(TooManyRedirectsError):
        await _client(handler).get("https://example.com/", ssrf_protect=True, max_redirects=3)


@pytest.mark.asyncio
async def test_oversized_streaming_body_aborted_before_full_read():
    produced = {"chunks": 0}

    async def body():
        for _ in range(10_000):  # ~10 MB if fully consumed
            produced["chunks"] += 1
            yield b"x" * 1024

    def handler(req):
        return httpx.Response(200, content=body(), headers={"content-type": "text/html"})

    with pytest.raises(ResponseTooLargeError, match="exceeds max limit"):
        await _client(handler).get("https://example.com/", ssrf_protect=True, max_bytes=8 * 1024)
    assert produced["chunks"] < 50  # aborted promptly, not buffered


@pytest.mark.asyncio
async def test_declared_content_length_over_limit_rejected_without_reading():
    def handler(req):
        return httpx.Response(200, headers={"content-length": "999999999", "content-type": "text/html"}, content=b"x")

    with pytest.raises(ResponseTooLargeError):
        await _client(handler).get("https://example.com/", ssrf_protect=True, max_bytes=1024)


@pytest.mark.asyncio
async def test_ssrf_errors_are_not_retried():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(302, headers={"Location": "http://10.0.0.1/"})

    client = SafeAsyncHttpClient(
        max_retries=3, min_request_interval_seconds=0.0,
        transport=httpx.MockTransport(handler), resolver=_resolver(PUBLIC),
    )
    with pytest.raises(SSRFValidationError):
        await client.get("https://example.com/", ssrf_protect=True)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_web_fetcher_uses_hardened_path_and_blocks_private_redirect():
    from connectors.web_fetcher import WebFetcher

    def handler(req):
        return httpx.Response(302, headers={"Location": "http://192.168.1.10/secret"})

    fetcher = WebFetcher(http_client=_client(handler))
    fetcher.validate_url_security = lambda url: None  # skip real DNS for the first hop
    with pytest.raises(SSRFValidationError):
        await fetcher.fetch("https://example.com/page")


# ───────────────────────────── Reranker reuse ────────────────────────────────

def test_shared_reranker_is_one_instance_per_config():
    from ranking.ranking_service import RankingService, get_shared_reranker

    a = RankingService().reranker
    b = RankingService().reranker
    assert a is b is get_shared_reranker()


def test_model_loaded_once_under_concurrency():
    from ranking.cross_encoder_reranker import CrossEncoderReranker

    constructed = []

    class FakeCE:
        def __init__(self, *a, **k):
            constructed.append(1)

        def predict(self, pairs, batch_size=16, show_progress_bar=False):
            return [float(len(p[1])) for p in pairs]

    rr = CrossEncoderReranker(model_name="fake", fallback_mode="hybrid_score")
    with patch.dict("sys.modules", {"sentence_transformers": SimpleNamespace(CrossEncoder=FakeCE)}):
        results: list[list[float]] = []
        threads = [
            threading.Thread(target=lambda: results.append(rr.score_pairs("q", ["a", "bb", "ccc"])))
            for _ in range(8)
        ]
        [t.start() for t in threads]
        [t.join() for t in threads]
    assert len(constructed) == 1
    assert all(r == [1.0, 2.0, 3.0] for r in results)  # deterministic scores preserved


def test_failed_load_uses_fallback_and_is_not_retried_within_cooldown():
    from ranking.cross_encoder_reranker import CrossEncoderReranker

    attempts = []

    class BoomCE:
        def __init__(self, *a, **k):
            attempts.append(1)
            raise OSError("weights missing at C:\\secret\\path")

    rr = CrossEncoderReranker(model_name="fake", fallback_mode="hybrid_score")
    with patch.dict("sys.modules", {"sentence_transformers": SimpleNamespace(CrossEncoder=BoomCE)}):
        assert rr.score_pairs("q", ["a"]) == [0.0]
        assert rr.score_pairs("q", ["a"]) == [0.0]
    assert len(attempts) == 1
    assert "secret" not in (rr._load_error_msg or "")  # only the exception class is kept


# ───────────────────────────── Redaction / errors ────────────────────────────

def test_redact_secrets_and_paths(monkeypatch):
    from core.config import get_settings
    from core.paths import REPO_ROOT
    from core.redaction import redact

    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-gemini-value-123")
    get_settings.cache_clear()
    txt = (
        f"failed key={_FAKE_GOOGLE_KEY} token: abc123def456 "
        "Authorization: Bearer abcdefghijklmnop super-secret-gemini-value-123 "
        "postgresql://user:hunter2pw@localhost:5433/db "
        f"at {REPO_ROOT}/backend/x.py"
    )
    out = redact(txt)
    for leaked in ("AIzaSy", "abc123def456", "abcdefghijklmnop", "super-secret", "hunter2pw"):
        assert leaked not in out
    assert "<repo>" in out and str(REPO_ROOT) not in out


@pytest.fixture
def raw_client(mock_infra):
    from db import database as db_module
    from main import app

    cm = [
        patch.object(db_module.db_manager, "connect", new_callable=AsyncMock),
        patch.object(db_module.db_manager, "disconnect", new_callable=AsyncMock),
    ]
    [c.start() for c in cm]
    yield AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t")
    [c.stop() for c in cm]


ANSWER_BODY = {"query": "what is RAG"}
LEAKY = (
    f"Gemini generation failed: 429 RESOURCE_EXHAUSTED quota key={_FAKE_GOOGLE_KEY} "
    "at C:\\Users\\victim\\project\\secret.py"
)


async def _post_answer(client, exc):
    with patch(
        "synthesis.answer_service.ResearchAnswerService.answer_question",
        new_callable=AsyncMock,
        side_effect=exc,
    ):
        return await client.post("/api/v1/research/answer", json=ANSWER_BODY)


@pytest.mark.asyncio
async def test_provider_rate_limit_is_429_and_sanitized(raw_client):
    from llm.models import LLMGenerationError

    r = await _post_answer(raw_client, LLMGenerationError(LEAKY))
    assert r.status_code == 429
    body = r.json()
    assert body["code"] == "llm_rate_limited" and body["request_id"]
    text = r.text
    for leaked in ("AIza", "victim", "secret.py", "RESOURCE_EXHAUSTED"):
        assert leaked not in text


@pytest.mark.asyncio
async def test_provider_categories_map_to_stable_codes(raw_client):
    from llm.models import LLMConfigurationError, LLMGenerationError

    cases = [
        (LLMGenerationError("Gemini generation failed: 404 models/x is not found"), 502, "llm_model_unavailable"),
        (LLMGenerationError("Gemini generation failed: 403 API key not valid"), 502, "llm_auth_error"),
        (LLMGenerationError("Gemini generation failed: Deadline exceeded"), 504, "llm_timeout"),
        (LLMGenerationError("Gemini generation failed: weird"), 502, "llm_provider_error"),
        (LLMConfigurationError("GEMINI_API_KEY is not configured"), 503, "llm_not_configured"),
    ]
    for exc, status, code in cases:
        r = await _post_answer(raw_client, exc)
        assert (r.status_code, r.json()["code"]) == (status, code)


@pytest.mark.asyncio
async def test_unhandled_exception_is_generic_500_without_detail(raw_client):
    r = await _post_answer(raw_client, RuntimeError("password=hunter2 /home/user/.env sqlalchemy dsn"))
    assert r.status_code == 500
    assert r.json()["code"] == "internal_error"
    assert "hunter2" not in r.text and "/home/user" not in r.text and "sqlalchemy" not in r.text


@pytest.mark.asyncio
async def test_validation_errors_keep_field_details_without_echoing_input(raw_client):
    r = await raw_client.post("/api/v1/research/answer", json={"query": "", "evidence_k": 999})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "validation_error"
    fields = {d["field"] for d in body["detail"]}
    assert {"query", "evidence_k"} <= fields
    assert "999" not in r.text


@pytest.mark.asyncio
async def test_app_exceptions_keep_actionable_codes(raw_client):
    from retrieval.index_manifest import ManifestMissingError

    r = await _post_answer(raw_client, ManifestMissingError("No active vector index exists."))
    assert r.status_code == 503 and r.json()["code"] == "index_manifest_missing"


@pytest.mark.asyncio
async def test_web_fetch_ssrf_returns_generic_400(raw_client):
    r = await raw_client.get("/api/v1/sources/web/fetch", params={"url": "http://10.0.0.1/admin"})
    assert r.status_code == 400
    assert "10.0.0.1" not in r.text
    assert "Security check failed" in r.json()["detail"]


# ───────────────────────────── Client lifecycle ──────────────────────────────

@pytest.mark.asyncio
async def test_retriever_aclose_closes_clients():
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.hybrid_retriever import HybridRetriever

    os_client, qd_client = AsyncMock(), AsyncMock()
    h = HybridRetriever(
        bm25_retriever=BM25Retriever(client=os_client),
        dense_retriever=DenseRetriever(client=qd_client),
    )
    await h.aclose()
    os_client.close.assert_awaited_once()
    qd_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_aclose_services_closes_and_clears_registry():
    from api import deps

    svc = SimpleNamespace(aclose=AsyncMock())
    deps._instances["x"] = svc
    await deps.aclose_services()
    svc.aclose.assert_awaited_once()
    assert deps._instances == {}
    await deps.aclose_services()  # idempotent


def test_api_services_are_shared_not_per_request():
    from api import deps

    assert deps.get_answer_service() is deps.get_answer_service()
    assert deps.get_hybrid_retriever() is deps.get_retrieval_service().hybrid_retriever
    assert deps.get_answer_service().retrieval_service is deps.get_retrieval_service()


@pytest.mark.asyncio
async def test_indexing_service_aclose():
    from embedding.mock_provider import MockEmbeddingProvider
    from retrieval.indexing_service import IndexingService

    os_c, qd_c = AsyncMock(), AsyncMock()
    svc = IndexingService(
        opensearch_client=os_c, qdrant_client=qd_c, embedding_provider=MockEmbeddingProvider(64)
    )
    await svc.aclose()
    os_c.close.assert_awaited_once()
    qd_c.close.assert_awaited_once()


# ───────────────────────────── Readiness semantics ───────────────────────────

@pytest.mark.asyncio
async def test_ready_when_only_optional_services_are_absent(mock_index, monkeypatch):
    """Real index + configured LLM + reachable DB => 200, even with no OpenSearch/Redis."""
    from core.config import get_settings
    from db import database as db_module
    from main import app

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()
    with patch("api.v1.health._check_postgres", new_callable=AsyncMock) as pg, patch.object(
        db_module.db_manager, "connect", new_callable=AsyncMock
    ), patch.object(db_module.db_manager, "disconnect", new_callable=AsyncMock):
        from api.v1.health import CheckResult

        pg.return_value = CheckResult(status="ok")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get("/api/v1/ready")
    body = r.json()
    assert r.status_code == 200, body
    assert body["checks"]["index"]["status"] == "ok"
    assert body["checks"]["opensearch"]["status"] == "skipped"
    assert body["checks"]["redis"]["status"] == "skipped"


@pytest.mark.asyncio
async def test_check_index_reports_genuine_failures(tmp_path, monkeypatch):
    from api.v1.health import _check_index

    # no index at all
    res = await _check_index()
    assert res.status == "error" and res.error == "index_manifest_missing"


@pytest.mark.asyncio
async def test_check_index_detects_embedding_model_mismatch(mock_index, monkeypatch):
    from api.v1.health import _check_index
    from core.config import get_settings

    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")  # index was built with mock vectors
    get_settings.cache_clear()
    res = await _check_index()
    assert res.status == "error" and res.error == "embedding_model_mismatch"


@pytest.mark.asyncio
async def test_check_llm_configuration(monkeypatch):
    from api.v1.health import _check_llm
    from core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    res = await _check_llm()
    assert res.status == "error" and res.error == "gemini_api_key_not_configured"
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 20)
    get_settings.cache_clear()
    assert (await _check_llm()).status == "ok"
    assert "x" * 20 not in (await _check_llm()).model_dump_json()


@pytest.mark.asyncio
async def test_optional_probe_failures_are_informational(monkeypatch):
    from api.v1.health import CheckResult, _optional
    from core.config import get_settings

    monkeypatch.setenv("READINESS_PROBE_OPTIONAL", "true")
    get_settings.cache_clear()
    probe = AsyncMock(return_value=CheckResult(status="error", error="ConnectionRefusedError"))
    res = await _optional(probe, "Redis")
    assert res.status == "error" and res.required is False
    probe.assert_awaited_once()


# ───────────────────── Claim / verification consistency ──────────────────────

class _FakeLLM:
    provider_name, model_name = "fake", "fake-1"

    def __init__(self, payload: dict):
        self.payload = payload

    async def generate(self, **kw):
        import json

        from llm.models import LLMResponse

        return LLMResponse(content=json.dumps(self.payload), provider="fake", model="fake-1")


def _evidence_pack():
    from ranking.evidence import EvidencePack, EvidencePassage

    ps = [
        EvidencePassage(
            passage_id=f"p{i}", document_id=f"d{i}", text=f"passage {i} text", title=f"Doc {i}",
            source_type="arxiv", source_id=f"2000.0000{i}", url=f"https://arxiv.org/abs/2000.0000{i}",
            relevance_score=1.0 - i / 10, rank=i, metadata={"authors": ["A. Author"]},
        )
        for i in (1, 2)
    ]
    return EvidencePack(query="q", selected_passages=ps, total_candidates=2, total_evidence=2)


@pytest.mark.asyncio
async def test_section_claims_share_verified_status_and_citations():
    from synthesis.answer_service import ResearchAnswerRequest, ResearchAnswerService

    payload = {
        "summary": "S",
        "insufficient_evidence": False,
        "claims": [
            {"claim_id": "claim_001", "text": "Alpha is true.", "evidence_ids": ["evidence_001"]},
            {"claim_id": "claim_002", "text": "Beta cites a ghost.", "evidence_ids": ["evidence_099"]},
        ],
        "sections": [
            {
                "title": "Findings",
                "claims": [
                    # same claim, different whitespace/case and an extra evidence id
                    {"claim_id": "claim_001", "text": "alpha   is TRUE.", "evidence_ids": ["evidence_002"]},
                    # section-only claim reusing an existing id
                    {"claim_id": "claim_001", "text": "Gamma only in a section.", "evidence_ids": ["evidence_002"]},
                    # section-only claim with no evidence
                    {"claim_id": "claim_007", "text": "Delta has no evidence.", "evidence_ids": []},
                ],
            }
        ],
        "limitations": [],
    }
    retrieval = SimpleNamespace(
        retrieve_evidence=AsyncMock(
            return_value=SimpleNamespace(
                evidence_pack=_evidence_pack(),
                latency_breakdown=SimpleNamespace(total_retrieval_latency_ms=1.0),
            )
        )
    )
    svc = ResearchAnswerService(
        retrieval_service=retrieval, planning_service=object(), llm_provider=_FakeLLM(payload)
    )
    ans = await svc.answer_question(ResearchAnswerRequest(query="q", enable_decomposition=False))

    by_text = {c.text: c for c in ans.claims}
    assert len(ans.claims) == 4  # 2 top-level + 2 section-only claims appended
    ids = [c.claim_id for c in ans.claims]
    assert len(set(ids)) == len(ids)  # no id collisions

    alpha = by_text["Alpha is true."]
    assert alpha.support_status.value == "SUPPORTED"
    assert set(alpha.evidence_ids) == {"evidence_001", "evidence_002"}  # merged, not dropped
    assert by_text["Beta cites a ghost."].support_status.value == "UNSUPPORTED"
    assert by_text["Delta has no evidence."].support_status.value == "UNSUPPORTED"

    # Every section claim equals its top-level counterpart (status + citations)
    top = {c.claim_id: c for c in ans.claims}
    for sec in ans.sections:
        for sc in sec.claims:
            t = top[sc.claim_id]
            assert (sc.support_status, sc.citation_ids, sc.evidence_ids) == (
                t.support_status, t.citation_ids, t.evidence_ids,
            )
    gamma = next(c for c in ans.sections[0].claims if c.text.startswith("Gamma"))
    assert gamma.citation_ids and gamma.support_status.value == "SUPPORTED"

    # Honest verification level
    v = ans.verification
    assert v.verification_method == "structural_evidence_id_check"
    assert v.semantic_entailment_checked is False
    assert v.valid is False  # unsupported claims present


# ───────────────────── Gemini validation script (control flow) ───────────────

def _load_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "gemini_e2e_validate.py"
    spec = importlib.util.spec_from_file_location("gemini_e2e_validate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.mark.asyncio
async def test_gemini_script_not_run_without_key(monkeypatch):
    from core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    rep = await _load_script().validate(get_settings())
    assert rep["status"] == "NOT_RUN"
    assert "GEMINI_API_KEY" in rep["reason"]


@pytest.mark.asyncio
async def test_gemini_script_flow_with_injected_provider_is_labelled_not_real(mock_index, monkeypatch):
    from core.config import get_settings
    from llm.mock_provider import MockLLMProvider
    from ranking.cross_encoder_reranker import CrossEncoderReranker

    # avoid loading real weights in a unit test; the reranker falls back
    monkeypatch.setattr(CrossEncoderReranker, "_load_model", lambda self: None)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    get_settings.cache_clear()
    rep = await _load_script().validate(get_settings(), llm_provider=MockLLMProvider())
    assert rep["real_gemini_call"] is False
    assert rep["mode"].startswith("INJECTED_TEST_PROVIDER")
    assert rep["status"] != "PASS"  # a mock flow can never be reported as a Gemini pass
    assert rep["llm_calls"] == 1
    assert rep["checks"]["evidence_retrieved"] and rep["checks"]["plan_present"]
    assert rep["checks"]["neural_reranker_active"] is False  # honestly reported


@pytest.mark.asyncio
async def test_gemini_script_reports_quota_failure_without_retry_or_mock(mock_index, monkeypatch):
    from core.config import get_settings
    from llm.models import LLMGenerationError
    from ranking.cross_encoder_reranker import CrossEncoderReranker

    class Quota:
        provider_name, model_name = "gemini", "m"
        calls = 0

        async def generate(self, **kw):
            Quota.calls += 1
            raise LLMGenerationError(f"Gemini generation failed: 429 quota exceeded key={_FAKE_GOOGLE_KEY}")

    monkeypatch.setattr(CrossEncoderReranker, "_load_model", lambda self: None)
    get_settings.cache_clear()
    rep = await _load_script().validate(get_settings(), llm_provider=Quota())
    assert rep["status"] == "FAIL" and rep["stage"] == "generation"
    assert rep["error_category"] == "llm_rate_limited"
    assert Quota.calls == 1  # no retry
    assert "AIza" not in str(rep)


@pytest.mark.asyncio
async def test_truncated_llm_json_is_an_error_not_insufficient_evidence(raw_client):
    from llm.models import LLMResponse
    from synthesis.answer_service import ResearchAnswerRequest, ResearchAnswerService

    class Truncating:
        provider_name, model_name = "fake", "fake-1"

        async def generate(self, **kw):
            return LLMResponse(
                content='{"summary": "cut off mid-str', provider="fake", model="fake-1",
                finish_reason="MAX_TOKENS",
            )

    retrieval = SimpleNamespace(
        retrieve_evidence=AsyncMock(
            return_value=SimpleNamespace(
                evidence_pack=_evidence_pack(),
                latency_breakdown=SimpleNamespace(total_retrieval_latency_ms=1.0),
            )
        )
    )
    svc = ResearchAnswerService(
        retrieval_service=retrieval, planning_service=object(), llm_provider=Truncating()
    )
    from llm.models import LLMGenerationError

    with pytest.raises(LLMGenerationError, match="truncated or malformed"):
        await svc.answer_question(ResearchAnswerRequest(query="q", enable_decomposition=False))

    r = await _post_answer(
        raw_client, LLMGenerationError("Structured output could not be parsed (finish_reason=MAX_TOKENS)")
    )
    assert r.status_code == 502 and r.json()["code"] == "llm_malformed_output"


# ───────────── Phase 12: first-use model load must not block the event loop ─────────────

@pytest.mark.asyncio
async def test_dense_search_model_load_does_not_block_event_loop(mock_index):
    """A slow first model load (import + weights) runs off-loop, so the loop stays responsive."""
    import time

    from embedding.mock_provider import MockEmbeddingProvider
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.models import RetrievalQuery

    class SlowDim(MockEmbeddingProvider):
        @property
        def dimension(self) -> int:  # evaluated inside validate_for_read -> describe_embedding_provider
            time.sleep(0.6)  # simulates a blocking cold import / weight load
            return 384

    dense = DenseRetriever(embedding_provider=SlowDim(dimension=384), index_store=mock_index)
    dense._server_available = False

    ticks: list[float] = []

    async def heartbeat():
        for _ in range(8):
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.05)

    hb = asyncio.create_task(heartbeat())
    res = await dense.search(RetrievalQuery(query_text="dense retrieval", top_k=3))
    await hb
    assert res.metadata["embedding_is_mock"] is True
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert max(gaps) < 0.3, f"event loop was blocked for {max(gaps):.2f}s"

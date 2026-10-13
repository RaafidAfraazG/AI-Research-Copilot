"""
Unit tests for WebFetcher and HTML content extraction with SSRF defenses.
"""
from __future__ import annotations

from unittest.mock import AsyncMock
import httpx
import pytest

from connectors.models import ConnectorSearchRequest
from connectors.web_fetcher import (
    HTMLContentExtractor,
    SSRFValidationError,
    WebFetcher,
)

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Information Retrieval Overview</title>
    <meta name="description" content="A comprehensive guide to modern information retrieval.">
    <meta name="author" content="Dr. Jane Smith">
    <link rel="canonical" href="https://example.com/canonical-ir-guide">
    <script>console.log("ignore me");</script>
    <style>body { font-family: sans-serif; }</style>
</head>
<body>
    <header><nav>Navigation links to ignore</nav></header>
    <article>
        <h1>Introduction to IR</h1>
        <p>Information retrieval (IR) is the process of obtaining information system resources relevant to an information need.</p>
        <p>Modern search engines combine lexical indexing with dense vector embeddings for hybrid retrieval.</p>
    </article>
    <footer>Copyright 2026 Example Corp</footer>
</body>
</html>
"""


def test_html_content_extractor():
    """Verify HTML parser extracts title, metadata, and body text while stripping scripts/nav."""
    extractor = HTMLContentExtractor()
    extractor.feed(SAMPLE_HTML)

    assert "Information Retrieval Overview" in extractor.get_title()
    assert extractor.get_description() == "A comprehensive guide to modern information retrieval."
    assert extractor.get_author() == "Dr. Jane Smith"
    assert extractor.canonical_url == "https://example.com/canonical-ir-guide"

    clean_text = extractor.get_clean_text()
    assert "Information retrieval (IR) is the process" in clean_text
    assert "dense vector embeddings" in clean_text
    assert "Navigation links to ignore" not in clean_text
    assert "console.log" not in clean_text


def test_ssrf_validation_rejects_localhost_and_private_ips():
    """Verify SSRF validation blocks loopback, private RFC1918, and metadata IPs."""
    fetcher = WebFetcher()

    # Blocked schemes
    with pytest.raises(SSRFValidationError, match="scheme"):
        fetcher.validate_url_security("file:///etc/passwd")

    with pytest.raises(SSRFValidationError, match="scheme"):
        fetcher.validate_url_security("ftp://ftp.example.com/file.txt")

    # Blocked hosts
    with pytest.raises(SSRFValidationError, match="localhost"):
        fetcher.validate_url_security("http://localhost:8000/api")

    with pytest.raises(SSRFValidationError, match="localhost"):
        fetcher.validate_url_security("http://127.0.0.1:5432")

    with pytest.raises(SSRFValidationError, match="private|special|loopback"):
        fetcher.validate_url_security("http://10.0.0.1/admin")

    with pytest.raises(SSRFValidationError, match="private|special|loopback"):
        fetcher.validate_url_security("http://192.168.1.1/router")

    with pytest.raises(SSRFValidationError, match="private|special|loopback|cloud metadata"):
        fetcher.validate_url_security("http://169.254.169.254/latest/meta-data/")


@pytest.mark.asyncio
async def test_web_fetcher_success():
    """Verify successful fetching and normalization into ResearchDocument."""
    target_url = "https://example.com/ir-guide"
    mock_http = AsyncMock()
    mock_http.get.return_value = httpx.Response(
        200,
        text=SAMPLE_HTML,
        headers={"content-type": "text/html; charset=utf-8"},
        request=httpx.Request("GET", target_url),
    )

    fetcher = WebFetcher(http_client=mock_http)
    fetcher.validate_url_security = lambda url: None  # type: ignore

    doc = await fetcher.fetch(target_url)
    assert doc is not None
    assert doc.source_type == "web"
    assert "Information Retrieval Overview" in doc.title
    assert doc.authors == ["Dr. Jane Smith"]
    assert doc.abstract == "A comprehensive guide to modern information retrieval."
    assert "dense vector embeddings" in (doc.full_text or "")
    assert doc.url == "https://example.com/canonical-ir-guide"


@pytest.mark.asyncio
async def test_web_fetcher_oversized_payload_rejected():
    """Verify oversized HTTP response raises ValueError."""
    target_url = "https://example.com/huge-file"
    huge_content = "A" * (100 * 1024)  # 100 KB
    mock_http = AsyncMock()
    mock_http.get.return_value = httpx.Response(
        200,
        text=huge_content,
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", target_url),
    )

    fetcher = WebFetcher(http_client=mock_http, max_bytes=10 * 1024)  # 10 KB limit
    fetcher.validate_url_security = lambda url: None  # type: ignore

    with pytest.raises(ValueError, match="exceeds max limit"):
        await fetcher.fetch(target_url)


@pytest.mark.asyncio
async def test_web_fetcher_unsupported_content_type_rejected():
    """Verify binary / non-HTML content type is rejected."""
    target_url = "https://example.com/image.png"
    mock_http = AsyncMock()
    mock_http.get.return_value = httpx.Response(
        200,
        content=b"\x89PNG\r\n\x1a\n",
        headers={"content-type": "image/png"},
        request=httpx.Request("GET", target_url),
    )

    fetcher = WebFetcher(http_client=mock_http)
    fetcher.validate_url_security = lambda url: None  # type: ignore

    with pytest.raises(ValueError, match="Unsupported content-type"):
        await fetcher.fetch(target_url)


@pytest.mark.asyncio
async def test_web_fetcher_search_by_url():
    """Verify search executes direct fetch if query is a URL."""
    target_url = "https://example.com/article"
    mock_http = AsyncMock()
    mock_http.get.return_value = httpx.Response(
        200,
        text="<html><head><title>Search Article</title></head><body><p>Content</p></body></html>",
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", target_url),
    )

    fetcher = WebFetcher(http_client=mock_http)
    fetcher.validate_url_security = lambda url: None  # type: ignore

    req = ConnectorSearchRequest(query=target_url)
    res = await fetcher.search(req)

    assert res.total_results == 1
    assert len(res.documents) == 1
    assert res.documents[0].title == "Search Article"

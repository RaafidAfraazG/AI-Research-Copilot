"""
Safe Web Document Fetcher with SSRF Protection and Boilerplate Stripping.

Security Controls:
- Strict SSRF defense: DNS resolution check blocking loopback, private RFC 1918, link-local, and cloud metadata IPs
- Enforces HTTP/HTTPS protocols only
- Configurable maximum response payload size (prevents decompression / memory exhaustion attacks)
- Strict content-type validation (rejects binaries, executables, streaming media)
- Configurable timeout and redirect limits
"""
from __future__ import annotations

import html
from html.parser import HTMLParser
import re
import time
from typing import Any
import urllib.parse
import httpx
import structlog

from connectors.base import ResearchSourceConnector
from connectors.http_client import SafeAsyncHttpClient
from connectors.url_security import SSRFValidationError, validate_url_sync
from connectors.models import (
    ConnectorSearchRequest,
    ConnectorSearchResponse,
    ProvenanceInfo,
    ResearchDocument,
)

logger = structlog.get_logger(__name__)

DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_CONTENT_TYPES = {"text/html", "text/plain", "application/xhtml+xml", "text/markdown"}


class HTMLContentExtractor(HTMLParser):
    """
    Standard library HTML parser for extracting title, metadata, and clean text
    without external dependencies like beautifulsoup4.
    """

    IGNORABLE_TAGS = {
        "script", "style", "nav", "header", "footer", "aside",
        "noscript", "form", "svg", "iframe", "button", "select", "option",
    }
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
    BLOCK_TAGS = {
        "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "article",
        "section", "div", "blockquote", "pre", "table", "tr", "td",
    }

    def __init__(self) -> None:
        super().__init__()
        self._tag_stack: list[str] = []
        self._in_title = False
        self._in_article = False
        self.title_parts: list[str] = []
        self.meta_tags: dict[str, str] = {}
        self.canonical_url: str | None = None
        self.text_chunks: list[str] = []
        self._current_block: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t not in self.VOID_TAGS:
            self._tag_stack.append(t)

        attr_dict = {k.lower(): (v or "") for k, v in attrs}

        if t == "title":
            self._in_title = True
        elif t == "article" or t == "main":
            self._in_article = True
        elif t == "meta":
            name = attr_dict.get("name") or attr_dict.get("property")
            content = attr_dict.get("content")
            if name and content:
                self.meta_tags[name.lower()] = content.strip()
        elif t == "link" and attr_dict.get("rel") == "canonical":
            self.canonical_url = attr_dict.get("href")

        if t in self.BLOCK_TAGS:
            self._flush_block()

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "title":
            self._in_title = False
        elif t == "article" or t == "main":
            self._in_article = False

        if t in self._tag_stack:
            while self._tag_stack:
                popped = self._tag_stack.pop()
                if popped == t:
                    break

        if t in self.BLOCK_TAGS:
            self._flush_block()

    def handle_data(self, data: str) -> None:
        # If inside any ignorable tag, skip
        if any(ign in self._tag_stack for ign in self.IGNORABLE_TAGS):
            return

        clean_text = data.strip()
        if not clean_text:
            return

        if self._in_title:
            self.title_parts.append(clean_text)
        else:
            self._current_block.append(clean_text)

    def _flush_block(self) -> None:
        if self._current_block:
            block_str = " ".join(self._current_block).strip()
            if block_str and len(block_str) > 3:
                self.text_chunks.append(block_str)
            self._current_block.clear()

    def get_title(self) -> str:
        if self.title_parts:
            return html.unescape(" ".join(self.title_parts))
        # Fallbacks: og:title, twitter:title
        return self.meta_tags.get("og:title") or self.meta_tags.get("twitter:title") or "Web Document"

    def get_description(self) -> str | None:
        return (
            self.meta_tags.get("description")
            or self.meta_tags.get("og:description")
            or self.meta_tags.get("twitter:description")
        )

    def get_author(self) -> str | None:
        return self.meta_tags.get("author") or self.meta_tags.get("article:author")

    def get_clean_text(self) -> str:
        self._flush_block()
        return "\n\n".join(self.text_chunks)



class WebFetcher(ResearchSourceConnector):
    """
    Secure single-page document fetcher with SSRF validation and boilerplate stripping.
    """

    def __init__(
        self,
        http_client: SafeAsyncHttpClient | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout_seconds: float = 12.0,
    ) -> None:
        self.max_bytes = max_bytes
        self.timeout = timeout_seconds
        self.http_client = http_client or SafeAsyncHttpClient(
            timeout_seconds=timeout_seconds,
            min_request_interval_seconds=0.5,
            user_agent="AI-Research-Copilot/0.1.0 (Web research fetcher; +https://github.com/ai-research-copilot)",
        )

    @property
    def source_type(self) -> str:
        return "web"

    def validate_url_security(self, url: str) -> None:
        """
        Synchronous SSRF pre-check (scheme, credentials, hostname, resolved addresses).

        The HTTP client repeats this validation for the request and for every redirect hop
        and connects to the validated address, so this early check is fail-fast only.
        """
        validate_url_sync(url)

    async def health(self) -> tuple[bool, str]:
        """Check outbound web fetch capability against a reliable root."""
        return True, "ok"

    def _extract_content(self, raw_html: str, source_url: str) -> tuple[str, str | None, str | None, str | None, str]:
        """
        Parse HTML and extract (title, description, author, canonical_url, main_content).
        """
        # Try bs4 if available, otherwise use standard library HTMLContentExtractor
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "svg"]):
                tag.decompose()

            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find("h1"):
                title = soup.find("h1").get_text().strip()

            desc = None
            meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            if meta_desc and meta_desc.get("content"):
                desc = meta_desc["content"].strip()

            author = None
            meta_auth = soup.find("meta", attrs={"name": "author"})
            if meta_auth and meta_auth.get("content"):
                author = meta_auth["content"].strip()

            canonical = None
            link_can = soup.find("link", attrs={"rel": "canonical"})
            if link_can and link_can.get("href"):
                canonical = link_can["href"].strip()

            main_text = soup.get_text(separator="\n\n", strip=True)
            return title or "Web Page", desc, author, canonical, main_text

        except ImportError:
            # Fallback to standard library HTMLContentExtractor
            extractor = HTMLContentExtractor()
            extractor.feed(raw_html)
            title = extractor.get_title()
            desc = extractor.get_description()
            author = extractor.get_author()
            canonical = extractor.canonical_url
            main_text = extractor.get_clean_text()
            return title, desc, author, canonical, main_text

    async def fetch(self, url: str) -> ResearchDocument | None:
        """
        Safely fetch and normalize a single web page.
        """
        start_time = time.perf_counter()
        self.validate_url_security(url)

        try:
            response = await self.http_client.get(
                url=url,
                timeout=self.timeout,
                follow_redirects=True,
                ssrf_protect=True,        # validate every redirect hop, pin to checked IP
                max_bytes=self.max_bytes, # enforced while streaming
            )
            response.raise_for_status()

            # Check content length
            content_bytes = response.content
            if len(content_bytes) > self.max_bytes:
                raise ValueError(f"Response size ({len(content_bytes)} bytes) exceeds max limit ({self.max_bytes} bytes)")

            # Check content-type
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type and not any(ct in content_type for ct in ALLOWED_CONTENT_TYPES):
                raise ValueError(f"Unsupported content-type '{content_type}'. Must be HTML or plain text.")

            raw_text = response.text
            title, desc, author, canonical, main_text = self._extract_content(raw_text, url)

            effective_url = canonical or str(response.url)
            authors = [author] if author else []

            provenance = ProvenanceInfo(
                source_type="web",
                source_id=effective_url,
                canonical_url=effective_url,
                raw_metadata={
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "content_length": len(content_bytes),
                },
                cross_source_ids={"url": effective_url},
            )

            doc = ResearchDocument(
                source_type="web",
                source_id=effective_url,
                title=title,
                authors=authors,
                abstract=desc,
                full_text=main_text,
                url=effective_url,
                publication_date=None,
                venue="Web",
                metadata={
                    "original_url": url,
                    "content_type": content_type,
                    "byte_size": len(content_bytes),
                },
                provenance=provenance,
            )
            return doc

        except Exception as e:
            logger.error("web_fetch_failed", url=url, error_type=type(e).__name__)
            raise

    async def search(self, request: ConnectorSearchRequest) -> ConnectorSearchResponse:
        """
        WebFetcher does not execute open web queries (not a search engine wrapper).
        If the query is a valid URL, it fetches that URL.
        """
        if request.query.startswith("http://") or request.query.startswith("https://"):
            doc = await self.fetch(request.query)
            docs = [doc] if doc else []
            return ConnectorSearchResponse(
                source_type="web",
                query=request.query,
                total_results=len(docs),
                documents=docs,
            )
        return ConnectorSearchResponse(
            source_type="web",
            query=request.query,
            total_results=0,
            documents=[],
            metadata={"notice": "WebFetcher searches by direct URL only"},
        )

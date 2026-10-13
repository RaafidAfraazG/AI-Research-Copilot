"""
arXiv Research Source Connector.
Fetches research papers and abstracts from the official arXiv REST/Atom API.
"""
from __future__ import annotations

import re
import time
from typing import Any
import xml.etree.ElementTree as ET
import structlog

from connectors.base import ResearchSourceConnector
from connectors.http_client import SafeAsyncHttpClient
from connectors.models import (
    CitationInfo,
    ConnectorSearchRequest,
    ConnectorSearchResponse,
    ProvenanceInfo,
    ResearchDocument,
)

logger = structlog.get_logger(__name__)

ARXIV_API_BASE_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


class ArxivConnector(ResearchSourceConnector):
    """
    Connector for searching and fetching papers from arXiv via the official Atom API.
    """

    def __init__(
        self,
        http_client: SafeAsyncHttpClient | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        # arXiv recommends at least 3 seconds between requests in high volume,
        # here we use SafeAsyncHttpClient with a 1.0s minimum interval for normal queries.
        self.http_client = http_client or SafeAsyncHttpClient(
            timeout_seconds=timeout_seconds,
            min_request_interval_seconds=1.0,
            user_agent="AI-Research-Copilot/0.1.0 (arXiv connector; mailto:research@copilot.local)",
        )

    @property
    def source_type(self) -> str:
        return "arxiv"

    async def health(self) -> tuple[bool, str]:
        """Verify connectivity to arXiv API."""
        try:
            res = await self.http_client.get(
                ARXIV_API_BASE_URL,
                params={"search_query": "all:electron", "max_results": 1},
                timeout=5.0,
            )
            if res.status_code == 200:
                return True, "ok"
            return False, f"HTTP {res.status_code}"
        except Exception as e:
            return False, str(e)

    def _clean_text(self, text: str | None) -> str:
        if not text:
            return ""
        # Collapse multiple whitespaces and newlines into single spaces
        return " ".join(text.split())

    def _extract_arxiv_id(self, raw_id: str) -> str:
        """
        Extract clean arXiv ID from URL or raw string.
        e.g. 'http://arxiv.org/abs/2005.11401v1' -> '2005.11401'
        """
        if not raw_id:
            return ""
        match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", raw_id)
        if match:
            return match.group(1)
        # Old style arXiv IDs: cs/0112017 or math.GT/0309136
        old_match = re.search(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})", raw_id)
        if old_match:
            return old_match.group(1)
        return raw_id.strip()

    def _parse_atom_entry(self, entry: ET.Element) -> ResearchDocument | None:
        """Parse a single <entry> element into a normalized ResearchDocument."""
        try:
            raw_id = entry.findtext(f"{ATOM_NS}id") or ""
            arxiv_id = self._extract_arxiv_id(raw_id)
            if not arxiv_id:
                return None

            title_elem = entry.find(f"{ATOM_NS}title")
            title = self._clean_text(title_elem.text if title_elem is not None else "")

            summary_elem = entry.find(f"{ATOM_NS}summary")
            abstract = self._clean_text(summary_elem.text if summary_elem is not None else "")

            published = entry.findtext(f"{ATOM_NS}published") or ""
            updated = entry.findtext(f"{ATOM_NS}updated") or ""

            # Extract authors
            authors: list[str] = []
            for author_elem in entry.findall(f"{ATOM_NS}author"):
                name_elem = author_elem.find(f"{ATOM_NS}name")
                if name_elem is not None and name_elem.text:
                    authors.append(self._clean_text(name_elem.text))

            # Extract categories
            categories: list[str] = []
            for cat_elem in entry.findall(f"{ATOM_NS}category"):
                term = cat_elem.get("term")
                if term:
                    categories.append(term)

            # Extract links
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url: str | None = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            for link_elem in entry.findall(f"{ATOM_NS}link"):
                rel = link_elem.get("rel")
                title_attr = link_elem.get("title")
                href = link_elem.get("href")
                if rel == "alternate" and href:
                    abs_url = href
                elif title_attr == "pdf" and href:
                    pdf_url = href

            # Extract journal ref, doi, comment
            doi = entry.findtext(f"{ARXIV_NS}doi")
            journal_ref = entry.findtext(f"{ARXIV_NS}journal_ref")
            comment = entry.findtext(f"{ARXIV_NS}comment")

            citation_info = CitationInfo(
                arxiv_id=arxiv_id,
                doi=doi,
            )

            provenance = ProvenanceInfo(
                source_type="arxiv",
                source_id=arxiv_id,
                canonical_url=abs_url,
                raw_metadata={
                    "published": published,
                    "updated": updated,
                    "journal_ref": journal_ref,
                    "comment": comment,
                },
                cross_source_ids={"arxiv": arxiv_id},
            )

            doc = ResearchDocument(
                source_type="arxiv",
                source_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                url=abs_url,
                pdf_url=pdf_url,
                publication_date=published[:10] if published else None,
                venue=journal_ref or "arXiv",
                categories=categories,
                citation_info=citation_info,
                metadata={
                    "arxiv_id": arxiv_id,
                    "journal_ref": journal_ref,
                    "comment": comment,
                    "doi": doi,
                },
                provenance=provenance,
            )
            return doc
        except Exception as e:
            logger.warning("arxiv_entry_parse_error", error=str(e))
            return None

    def _parse_atom_feed(self, xml_text: str) -> list[ResearchDocument]:
        """Parse full Atom feed XML string into a list of ResearchDocuments."""
        if not xml_text or not xml_text.strip():
            return []

        try:
            root = ET.fromstring(xml_text)
            documents: list[ResearchDocument] = []
            for entry in root.findall(f"{ATOM_NS}entry"):
                doc = self._parse_atom_entry(entry)
                if doc:
                    documents.append(doc)
            return documents
        except ET.ParseError as e:
            logger.error("arxiv_xml_parse_error", error=str(e))
            return []

    async def search(self, request: ConnectorSearchRequest) -> ConnectorSearchResponse:
        """
        Search arXiv by keyword query.
        """
        start_time = time.perf_counter()
        
        # Build arXiv query: all:term1 AND all:term2
        clean_q = request.query.strip()
        formatted_query = f"all:{clean_q}"

        params: dict[str, Any] = {
            "search_query": formatted_query,
            "start": request.offset,
            "max_results": request.limit,
            "sortBy": request.sort_by if request.sort_by in {"relevance", "lastUpdatedDate", "submittedDate"} else "relevance",
            "sortOrder": "descending",
        }

        try:
            response = await self.http_client.get(ARXIV_API_BASE_URL, params=params)
            response.raise_for_status()
            docs = self._parse_atom_feed(response.text)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "arxiv_search_completed",
                query=request.query,
                limit=request.limit,
                results_count=len(docs),
                latency_ms=latency_ms,
            )

            return ConnectorSearchResponse(
                source_type="arxiv",
                query=request.query,
                total_results=len(docs),
                documents=docs,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error("arxiv_search_failed", query=request.query, error=str(e), latency_ms=latency_ms)
            return ConnectorSearchResponse(
                source_type="arxiv",
                query=request.query,
                total_results=0,
                documents=[],
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )

    async def fetch(self, identifier: str) -> ResearchDocument | None:
        """
        Fetch a single arXiv paper by its arXiv ID or URL.
        """
        arxiv_id = self._extract_arxiv_id(identifier)
        if not arxiv_id:
            return None

        params = {"id_list": arxiv_id}
        try:
            response = await self.http_client.get(ARXIV_API_BASE_URL, params=params)
            response.raise_for_status()
            docs = self._parse_atom_feed(response.text)
            return docs[0] if docs else None
        except Exception as e:
            logger.error("arxiv_fetch_failed", arxiv_id=arxiv_id, error=str(e))
            return None

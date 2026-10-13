"""
Semantic Scholar Research Source Connector.
Queries the official Semantic Scholar Academic Graph API for papers, citation graphs, and TLDRs.
"""
from __future__ import annotations

import time
from typing import Any
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
from core.config import Settings, get_settings

logger = structlog.get_logger(__name__)

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = (
    "paperId,title,authors,abstract,year,publicationDate,venue,"
    "citationCount,referenceCount,influentialCitationCount,openAccessPdf,url,tldr,fieldsOfStudy,externalIds"
)


class SemanticScholarConnector(ResearchSourceConnector):
    """
    Connector for searching and fetching academic papers from Semantic Scholar.
    Works with or without an API key (handles public rate limits gracefully).
    """

    def __init__(
        self,
        api_key: str | None = None,
        http_client: SafeAsyncHttpClient | None = None,
        settings: Settings | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.api_key = api_key or self.settings.semantic_scholar_api_key
        
        # Free unauthenticated tier limit is ~100 req/5min.
        # Set 1.0s interval between calls to avoid 429s.
        self.http_client = http_client or SafeAsyncHttpClient(
            timeout_seconds=timeout_seconds,
            min_request_interval_seconds=1.0 if not self.api_key else 0.2,
            user_agent="AI-Research-Copilot/0.1.0 (Academic Graph API client)",
        )

    @property
    def source_type(self) -> str:
        return "semantic_scholar"

    def _get_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def health(self) -> tuple[bool, str]:
        """Check Semantic Scholar API reachability."""
        try:
            res = await self.http_client.get(
                f"{S2_BASE_URL}/paper/search",
                params={"query": "machine learning", "limit": 1, "fields": "title"},
                headers=self._get_headers(),
                timeout=5.0,
            )
            if res.status_code == 200:
                return True, "ok"
            return False, f"HTTP {res.status_code}"
        except Exception as e:
            return False, str(e)

    def _parse_paper_dict(self, data: dict[str, Any]) -> ResearchDocument | None:
        """Normalize a Semantic Scholar API paper dictionary into a ResearchDocument."""
        if not data or not isinstance(data, dict):
            return None

        paper_id = data.get("paperId") or data.get("id")
        title = (data.get("title") or "").strip()
        if not paper_id or not title:
            return None

        # Authors
        authors: list[str] = []
        for author in data.get("authors") or []:
            name = author.get("name") if isinstance(author, dict) else str(author)
            if name and name.strip():
                authors.append(name.strip())

        # Abstract
        abstract = data.get("abstract") or ""

        # Citation metrics & external IDs
        external_ids = data.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv") or external_ids.get("arxiv")
        doi = external_ids.get("DOI") or external_ids.get("doi")
        corpus_id = str(external_ids.get("CorpusId") or "")

        tldr_obj = data.get("tldr")
        tldr_text: str | None = None
        if isinstance(tldr_obj, dict):
            tldr_text = tldr_obj.get("text")
        elif isinstance(tldr_obj, str):
            tldr_text = tldr_obj

        citation_info = CitationInfo(
            citation_count=data.get("citationCount"),
            reference_count=data.get("referenceCount"),
            influential_citation_count=data.get("influentialCitationCount"),
            tldr=tldr_text,
            doi=doi,
            arxiv_id=arxiv_id,
            corpus_id=corpus_id if corpus_id else None,
        )

        # OpenAccess PDF & URL
        open_access_pdf = data.get("openAccessPdf") or {}
        pdf_url = open_access_pdf.get("url") if isinstance(open_access_pdf, dict) else None
        s2_url = data.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}"

        # Date & venue
        pub_date = data.get("publicationDate")
        if not pub_date and data.get("year"):
            pub_date = f"{data['year']}-01-01"

        venue = data.get("venue") or ""
        categories = data.get("fieldsOfStudy") or []

        cross_source_ids = {"semantic_scholar": paper_id}
        if arxiv_id:
            cross_source_ids["arxiv"] = str(arxiv_id)
        if doi:
            cross_source_ids["doi"] = str(doi)

        provenance = ProvenanceInfo(
            source_type="semantic_scholar",
            source_id=paper_id,
            canonical_url=s2_url,
            raw_metadata={"year": data.get("year"), "venue": venue, "externalIds": external_ids},
            cross_source_ids=cross_source_ids,
        )

        return ResearchDocument(
            source_type="semantic_scholar",
            source_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract if abstract else (tldr_text or None),
            url=s2_url,
            pdf_url=pdf_url,
            publication_date=pub_date,
            venue=venue,
            categories=[str(c) for c in categories if c],
            citation_info=citation_info,
            metadata={
                "paperId": paper_id,
                "externalIds": external_ids,
                "citationCount": data.get("citationCount"),
                "year": data.get("year"),
                "tldr": tldr_text,
            },
            provenance=provenance,
        )

    async def search(self, request: ConnectorSearchRequest) -> ConnectorSearchResponse:
        """
        Search Semantic Scholar Academic Graph API.
        """
        start_time = time.perf_counter()
        params: dict[str, Any] = {
            "query": request.query.strip(),
            "limit": request.limit,
            "offset": request.offset,
            "fields": DEFAULT_FIELDS,
        }
        if request.year:
            params["year"] = f"{request.year}-"

        try:
            response = await self.http_client.get(
                f"{S2_BASE_URL}/paper/search",
                params=params,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()
            raw_papers = data.get("data") or []
            total_found = data.get("total", len(raw_papers))

            docs: list[ResearchDocument] = []
            for item in raw_papers:
                doc = self._parse_paper_dict(item)
                if doc:
                    docs.append(doc)

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                "semantic_scholar_search_completed",
                query=request.query,
                limit=request.limit,
                results_count=len(docs),
                total_found=total_found,
                latency_ms=latency_ms,
            )

            return ConnectorSearchResponse(
                source_type="semantic_scholar",
                query=request.query,
                total_results=int(total_found),
                documents=docs,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error("semantic_scholar_search_failed", query=request.query, error=str(e), latency_ms=latency_ms)
            return ConnectorSearchResponse(
                source_type="semantic_scholar",
                query=request.query,
                total_results=0,
                documents=[],
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )

    async def fetch(self, identifier: str) -> ResearchDocument | None:
        """
        Fetch a paper by Semantic Scholar Paper ID, DOI (e.g. '10.1145/...'),
        or arXiv ID (e.g. 'ARXIV:2005.11401').
        """
        clean_id = identifier.strip()
        params = {"fields": DEFAULT_FIELDS}

        try:
            response = await self.http_client.get(
                f"{S2_BASE_URL}/paper/{clean_id}",
                params=params,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_paper_dict(data)
        except Exception as e:
            logger.error("semantic_scholar_fetch_failed", identifier=identifier, error=str(e))
            return None

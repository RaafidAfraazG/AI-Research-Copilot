"""
Document Deduplication and Cross-Source Normalization Engine.

Handles:
- URL normalization (strips tracking parameters, lowercases host, normalizes arXiv/DOI links)
- Canonical identity resolution across arXiv, Semantic Scholar, and Web sources
- Metadata merging without loss of provenance
- Content hash comparison
"""
from __future__ import annotations

import hashlib
import re
import urllib.parse
from typing import Any
import structlog

from connectors.models import ResearchDocument

logger = structlog.get_logger(__name__)

TRACKING_QUERY_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "source", "fbclid", "gclid", "msclkid", "_hsenc", "_hsmi",
}


def normalize_url(url: str | None) -> str:
    """
    Produce a canonical, tracking-free normalized URL string.
    """
    if not url or not url.strip():
        return ""

    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()

    # Normalize default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # Clean path (strip trailing slashes, resolve double slashes)
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")

    # Filter tracking query parameters and sort remaining
    clean_params: list[tuple[str, str]] = []
    if parsed.query:
        for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=False):
            if k.lower() not in TRACKING_QUERY_PARAMS:
                clean_params.append((k, v))
    clean_params.sort(key=lambda x: x[0])
    query_str = urllib.parse.urlencode(clean_params)

    # Reconstruct (omit fragment)
    canonical = f"{scheme}://{netloc}{path}"
    if query_str:
        canonical = f"{canonical}?{query_str}"
    return canonical


def extract_canonical_id(doc: ResearchDocument) -> str:
    """
    Extract a cross-source canonical identity key for a ResearchDocument.
    Prioritizes:
    1. arXiv ID (e.g. 'arxiv:2005.11401')
    2. DOI (e.g. 'doi:10.1145/3340531.3412007')
    3. Source type + source id (e.g. 'web:https://...')
    """
    # 1. Check citation_info and provenance for arXiv ID
    arxiv_id = (
        doc.citation_info.arxiv_id
        or (doc.provenance.cross_source_ids.get("arxiv") if doc.provenance else None)
        or (doc.source_id if doc.source_type == "arxiv" else None)
    )
    if arxiv_id:
        clean_arxiv = re.sub(r"v\d+$", "", str(arxiv_id).strip())
        return f"arxiv:{clean_arxiv}"

    # 2. Check for DOI
    doi = (
        doc.citation_info.doi
        or (doc.provenance.cross_source_ids.get("doi") if doc.provenance else None)
    )
    if doi:
        return f"doi:{str(doi).strip().lower()}"

    # 3. Check normalized URL
    norm_url = normalize_url(doc.url)
    if norm_url:
        return f"url:{norm_url}"

    # 4. Fallback to source_type:source_id
    return f"{doc.source_type}:{doc.source_id}"


class DocumentDeduplicator:
    """
    In-memory / repository-backed document deduplicator and metadata merger.
    """

    def __init__(self) -> None:
        # Maps canonical_key -> ResearchDocument
        self._seen_documents: dict[str, ResearchDocument] = {}
        # Maps content_hash -> canonical_key
        self._seen_hashes: dict[str, str] = {}

    def is_duplicate(self, doc: ResearchDocument) -> bool:
        """Check if document has already been processed."""
        key = extract_canonical_id(doc)
        if key in self._seen_documents:
            return True
        chash = doc.compute_content_hash()
        if chash in self._seen_hashes:
            return True
        return False

    def register_and_merge(self, doc: ResearchDocument) -> tuple[ResearchDocument, bool]:
        """
        Register a document or merge its metadata with an existing document.
        Returns (merged_document, is_new).
        """
        key = extract_canonical_id(doc)
        chash = doc.compute_content_hash()

        if key in self._seen_documents:
            existing = self._seen_documents[key]
            merged = self._merge_documents(existing, doc)
            self._seen_documents[key] = merged
            logger.info("merged_duplicate_document", canonical_key=key, source_types=[existing.source_type, doc.source_type])
            return merged, False

        if chash in self._seen_hashes:
            existing_key = self._seen_hashes[chash]
            existing = self._seen_documents[existing_key]
            merged = self._merge_documents(existing, doc)
            self._seen_documents[existing_key] = merged
            logger.info("merged_duplicate_content_hash", content_hash=chash)
            return merged, False

        # New document
        self._seen_documents[key] = doc
        self._seen_hashes[chash] = key
        return doc, True

    def _merge_documents(self, primary: ResearchDocument, secondary: ResearchDocument) -> ResearchDocument:
        """
        Merge secondary metadata into primary document without losing provenance.
        """
        # Pick best title (longer/more complete)
        title = primary.title if len(primary.title) >= len(secondary.title) else secondary.title

        # Merge authors (deduplicating)
        author_set = set(primary.authors)
        authors = list(primary.authors)
        for a in secondary.authors:
            if a not in author_set:
                authors.append(a)
                author_set.add(a)

        # Merge abstract / text
        abstract = primary.abstract or secondary.abstract
        full_text = primary.full_text or secondary.full_text

        # Merge citations
        citation_info = primary.citation_info.model_copy()
        if secondary.citation_info.citation_count is not None:
            if citation_info.citation_count is None or secondary.citation_info.citation_count > (citation_info.citation_count or 0):
                citation_info.citation_count = secondary.citation_info.citation_count
        if secondary.citation_info.tldr and not citation_info.tldr:
            citation_info.tldr = secondary.citation_info.tldr
        if secondary.citation_info.doi and not citation_info.doi:
            citation_info.doi = secondary.citation_info.doi

        # Merge categories
        categories = list(set(primary.categories + secondary.categories))

        # Merge provenance
        cross_ids = {}
        if primary.provenance:
            cross_ids.update(primary.provenance.cross_source_ids)
        if secondary.provenance:
            cross_ids.update(secondary.provenance.cross_source_ids)
        cross_ids[primary.source_type] = primary.source_id
        cross_ids[secondary.source_type] = secondary.source_id

        meta = primary.metadata.copy()
        meta.update(secondary.metadata)
        meta["merged_sources"] = list(cross_ids.keys())

        merged_provenance = primary.provenance.model_copy() if primary.provenance else None
        if merged_provenance:
            merged_provenance.cross_source_ids = cross_ids

        return primary.model_copy(
            update={
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "full_text": full_text,
                "categories": categories,
                "citation_info": citation_info,
                "metadata": meta,
                "provenance": merged_provenance,
            }
        )

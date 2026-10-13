"""
Citation Mapper service.
Maps evidence references into human-readable bibliographic citations and provenance objects.
"""
from __future__ import annotations

from ranking.evidence import EvidencePack, EvidencePassage
from synthesis.models import Citation, ClaimItem, SourceProvenance


class CitationMapper:
    """
    Transforms raw evidence IDs into structured, numbered bibliographic citations.
    """

    @classmethod
    def format_citation_string(
        cls,
        number: int,
        title: str,
        authors: list[str],
        source_type: str,
        source_id: str,
        publication_date: str | None = None,
        url: str | None = None,
    ) -> str:
        """Render standard academic citation string."""
        # Author string
        if authors:
            if len(authors) == 1:
                auth_str = authors[0]
            elif len(authors) == 2:
                auth_str = f"{authors[0]} & {authors[1]}"
            else:
                auth_str = f"{authors[0]} et al."
        else:
            auth_str = "Unknown Authors"

        # Year string
        year_str = ""
        if publication_date and len(publication_date) >= 4:
            year_str = f" ({publication_date[:4]})"

        # Source ref
        if source_type == "arxiv":
            source_ref = f", arXiv:{source_id}"
        elif source_type == "semantic_scholar":
            source_ref = f", Semantic Scholar ID:{source_id[:8]}"
        elif source_type == "web":
            source_ref = ""
        else:
            source_ref = f", {source_type}:{source_id}"

        url_str = f" <{url}>" if url else ""

        return f"[{number}] {auth_str}, \"{title}\"{year_str}{source_ref}.{url_str}"

    @classmethod
    def map_citations(
        cls,
        claims: list[ClaimItem],
        id_to_passage: dict[str, EvidencePassage],
    ) -> tuple[list[Citation], list[SourceProvenance]]:
        """
        Build ordered bibliographic citations and source provenances from claims and evidence mappings.
        Updates ClaimItem.citation_ids in-place.
        """
        evidence_to_citation_id: dict[str, str] = {}
        citations: list[Citation] = []
        sources_map: dict[str, SourceProvenance] = {}
        citation_counter = 1

        for claim in claims:
            claim_citation_ids: list[str] = []
            for ev_id in claim.evidence_ids:
                if ev_id not in id_to_passage:
                    continue

                if ev_id not in evidence_to_citation_id:
                    cite_id = f"cite_{citation_counter}"
                    evidence_to_citation_id[ev_id] = cite_id

                    passage = id_to_passage[ev_id]
                    authors = passage.metadata.get("authors") or []
                    pub_date = passage.metadata.get("publication_date") or passage.metadata.get("year")
                    doi = passage.metadata.get("doi")
                    arxiv_id = passage.metadata.get("arxiv_id")

                    formatted = cls.format_citation_string(
                        number=citation_counter,
                        title=passage.title,
                        authors=authors,
                        source_type=passage.source_type,
                        source_id=passage.source_id,
                        publication_date=str(pub_date) if pub_date else None,
                        url=passage.url,
                    )

                    citation = Citation(
                        citation_id=cite_id,
                        number=citation_counter,
                        title=passage.title,
                        source_type=passage.source_type,
                        source_id=passage.source_id,
                        authors=authors,
                        publication_date=str(pub_date) if pub_date else None,
                        url=passage.url,
                        doi=doi,
                        arxiv_id=arxiv_id,
                        evidence_id=ev_id,
                        formatted_citation=formatted,
                    )
                    citations.append(citation)
                    citation_counter += 1

                    # Track distinct source
                    src_key = f"{passage.source_type}:{passage.source_id}"
                    if src_key not in sources_map:
                        sources_map[src_key] = SourceProvenance(
                            source_id=passage.source_id,
                            source_type=passage.source_type,
                            title=passage.title,
                            url=passage.url,
                            evidence_count=1,
                        )
                    else:
                        sources_map[src_key].evidence_count += 1

                claim_citation_ids.append(evidence_to_citation_id[ev_id])

            claim.citation_ids = claim_citation_ids

        return citations, list(sources_map.values())

"""
Evidence selection and diversification engine.
Filters, deduplicates, and diversifies reranked passages to produce an EvidencePack.
"""
from __future__ import annotations

import re
import time
from typing import Any
import structlog

from core.config import Settings, get_settings
from ranking.evidence import EvidencePack, EvidencePassage
from retrieval.models import ScoredChunk

logger = structlog.get_logger(__name__)


def compute_token_jaccard(text_a: str, text_b: str) -> float:
    """Compute token Jaccard similarity between two passages for duplicate suppression."""
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


class EvidenceSelector:
    """
    Selects high-quality, diverse, non-redundant evidence passages from reranked candidates.
    """

    def __init__(
        self,
        max_evidence: int | None = None,
        min_score: float | None = None,
        max_per_document: int | None = None,
        similarity_threshold: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_evidence = max_evidence or self.settings.evidence_top_k
        self.min_score = min_score if min_score is not None else self.settings.evidence_min_score
        self.max_per_document = max_per_document or self.settings.evidence_max_per_document
        self.similarity_threshold = similarity_threshold or self.settings.evidence_similarity_threshold

    def select(
        self,
        query: str,
        reranked_candidates: list[ScoredChunk],
        max_evidence: int | None = None,
        min_score: float | None = None,
        max_per_document: int | None = None,
        similarity_threshold: float | None = None,
        retrieval_metadata: dict[str, Any] | None = None,
        ranking_metadata: dict[str, Any] | None = None,
    ) -> tuple[EvidencePack, float]:
        """
        Filter, deduplicate, and select top evidence passages.
        
        Returns:
            Tuple of (EvidencePack, latency_ms)
        """
        start_time = time.perf_counter()
        target_k = max_evidence or self.max_evidence
        score_thresh = min_score if min_score is not None else self.min_score
        doc_limit = max_per_document or self.max_per_document
        sim_thresh = similarity_threshold or self.similarity_threshold

        selected_passages: list[EvidencePassage] = []
        doc_counts: dict[str, int] = {}
        sources_seen: dict[str, int] = {}

        for chunk in reranked_candidates:
            # 1. Check max count
            if len(selected_passages) >= target_k:
                break

            # 2. Check minimum relevance threshold
            chunk_score = chunk.rerank_score if chunk.rerank_score is not None else chunk.score
            if score_thresh is not None and chunk_score < score_thresh:
                continue

            # 3. Check document diversity quota (unless no other documents available)
            doc_id = chunk.document_id
            curr_doc_count = doc_counts.get(doc_id, 0)
            if doc_limit is not None and curr_doc_count >= doc_limit:
                # If we still have plenty of candidates, skip to promote diversity
                continue

            # 4. Check near-duplicate redundancy against already selected passages
            is_redundant = False
            for existing in selected_passages:
                jaccard_sim = compute_token_jaccard(chunk.content, existing.text)
                if jaccard_sim >= sim_thresh:
                    is_redundant = True
                    break
            if is_redundant:
                continue

            # 5. Accept passage
            rank = len(selected_passages) + 1
            doc_counts[doc_id] = curr_doc_count + 1
            sources_seen[chunk.source_type] = sources_seen.get(chunk.source_type, 0) + 1

            passage = EvidencePassage(
                passage_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.content,
                title=chunk.title,
                source_type=chunk.source_type,
                source_id=chunk.source_id,
                url=chunk.url,
                relevance_score=round(chunk_score, 4),
                rank=rank,
                chunk_index=chunk.chunk_index,
                provenance={
                    "source_type": chunk.source_type,
                    "source_id": chunk.source_id,
                    "url": chunk.url,
                    "document_id": chunk.document_id,
                },
                metadata=chunk.metadata,
            )
            selected_passages.append(passage)

        # Build aggregated provenance
        aggregated_provenance = {
            "source_distribution": sources_seen,
            "distinct_documents": len(doc_counts),
            "source_ids": [p.source_id for p in selected_passages],
        }

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        evidence_pack = EvidencePack(
            query=query,
            selected_passages=selected_passages,
            total_candidates=len(reranked_candidates),
            total_evidence=len(selected_passages),
            retrieval_metadata=retrieval_metadata or {},
            ranking_metadata=ranking_metadata or {},
            provenance=aggregated_provenance,
        )

        logger.info(
            "evidence_selection_completed",
            query=query,
            candidates=len(reranked_candidates),
            selected=len(selected_passages),
            distinct_docs=len(doc_counts),
            latency_ms=latency_ms,
        )

        return evidence_pack, latency_ms

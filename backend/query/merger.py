"""
Evidence Merger and Deduplication Engine.
Combines evidence retrieved across multiple research subqueries into a unified,
deduplicated, diversified, and provenance-tracked MergedEvidencePack.
"""
from __future__ import annotations

import re
import time
from typing import Any
import structlog

from ranking.evidence import EvidencePassage
from ranking.evidence_selector import compute_token_jaccard
from query.models import (

    EvidenceCoverageMetrics,
    MergedEvidencePack,
    ResearchPlan,
    ResearchSubQuery,
)
from services.research_retrieval_service import ResearchRetrievalResponse

logger = structlog.get_logger(__name__)


class EvidenceMerger:
    """
    Merges, deduplicates, and balances evidence retrieved across multiple subqueries.
    """

    def __init__(
        self,
        max_total_evidence: int = 12,
        max_evidence_per_subquery: int = 4,
        similarity_threshold: float = 0.85,
    ) -> None:
        self.max_total_evidence = max_total_evidence
        self.max_evidence_per_subquery = max_evidence_per_subquery
        self.similarity_threshold = similarity_threshold

    def merge(
        self,
        plan: ResearchPlan,
        subquery_results: list[tuple[ResearchSubQuery, ResearchRetrievalResponse | None]],
        failed_subquery_ids: list[str] | None = None,
        planning_latency_ms: float = 0.0,
    ) -> MergedEvidencePack:
        """
        Merge multiple subquery retrieval outputs into a unified MergedEvidencePack.

        Args:
            plan: Original ResearchPlan.
            subquery_results: Pairs of (ResearchSubQuery, ResearchRetrievalResponse).
            failed_subquery_ids: List of subquery IDs that failed during execution.
            planning_latency_ms: Total latency spent in planning and retrieval.

        Returns:
            MergedEvidencePack with full provenance and coverage metrics.
        """
        start_time = time.perf_counter()
        failed_ids = set(failed_subquery_ids or [])

        # Subquery ID to candidate evidence passages
        subquery_candidates: dict[str, list[EvidencePassage]] = {}
        subquery_lookup: dict[str, ResearchSubQuery] = {sq.subquery_id: sq for sq in plan.subqueries}

        total_candidate_count = 0
        all_retrieval_meta: dict[str, Any] = {}
        all_ranking_meta: dict[str, Any] = {}

        for sq, resp in subquery_results:
            if resp is None:
                failed_ids.add(sq.subquery_id)
                continue

            total_candidate_count += resp.total_candidates
            selected = resp.evidence_pack.selected_passages
            subquery_candidates[sq.subquery_id] = selected[:self.max_evidence_per_subquery]

            all_retrieval_meta[sq.subquery_id] = resp.evidence_pack.retrieval_metadata
            all_ranking_meta[sq.subquery_id] = resp.evidence_pack.ranking_metadata

        # Deduplication and Interleaving (Round-Robin with priority)
        merged_passages: list[EvidencePassage] = []
        seen_passage_ids: set[str] = set()
        subquery_evidence_map: dict[str, list[str]] = {sq.subquery_id: [] for sq in plan.subqueries}

        # Sort subqueries by priority
        sorted_subqueries = sorted(plan.subqueries, key=lambda s: s.priority)

        # Round-robin selection across subqueries
        max_depth = max((len(cands) for cands in subquery_candidates.values()), default=0)

        for depth in range(max_depth):
            for sq in sorted_subqueries:
                sq_id = sq.subquery_id
                cands = subquery_candidates.get(sq_id, [])
                if depth >= len(cands):
                    continue

                passage = cands[depth]

                # Deduplication check by passage_id
                if passage.passage_id in seen_passage_ids:
                    # Update subquery evidence map for provenance traceability
                    subquery_evidence_map[sq_id].append(passage.passage_id)
                    # Also append subquery_id to existing passage provenance
                    for existing in merged_passages:
                        if existing.passage_id == passage.passage_id:
                            existing_subqueries = existing.provenance.setdefault("subquery_ids", [])
                            if sq_id not in existing_subqueries:
                                existing_subqueries.append(sq_id)
                    continue

                # Near-duplicate check by token overlap Jaccard
                is_near_dup = False
                for existing in merged_passages:
                    sim = compute_token_jaccard(passage.text, existing.text)
                    if sim >= self.similarity_threshold:
                        is_near_dup = True
                        subquery_evidence_map[sq_id].append(existing.passage_id)
                        existing_subqueries = existing.provenance.setdefault("subquery_ids", [])
                        if sq_id not in existing_subqueries:
                            existing_subqueries.append(sq_id)
                        break


                if is_near_dup:
                    continue

                # Add passage to merged list
                seen_passage_ids.add(passage.passage_id)

                # Clone and augment passage with subquery provenance
                cloned_provenance = dict(passage.provenance)
                cloned_provenance["subquery_id"] = sq_id
                cloned_provenance["subquery_text"] = sq.text
                cloned_provenance["subquery_purpose"] = sq.purpose
                cloned_provenance.setdefault("subquery_ids", [sq_id])

                merged_passage = EvidencePassage(
                    passage_id=passage.passage_id,
                    document_id=passage.document_id,
                    text=passage.text,
                    title=passage.title,
                    source_type=passage.source_type,
                    source_id=passage.source_id,
                    url=passage.url,
                    relevance_score=passage.relevance_score,
                    rank=len(merged_passages) + 1,
                    chunk_index=passage.chunk_index,
                    provenance=cloned_provenance,
                    metadata=dict(passage.metadata),
                )

                merged_passages.append(merged_passage)
                subquery_evidence_map[sq_id].append(merged_passage.passage_id)

                if len(merged_passages) >= self.max_total_evidence:
                    break

            if len(merged_passages) >= self.max_total_evidence:
                break

        # Re-sort merged passages by relevance score and normalize ranks
        merged_passages.sort(key=lambda p: p.relevance_score, reverse=True)
        for rank_idx, p in enumerate(merged_passages, start=1):
            p.rank = rank_idx

        # Calculate coverage metrics
        total_subqueries = len(plan.subqueries)
        subqueries_with_evidence = sum(
            1 for sq_id, p_ids in subquery_evidence_map.items() if len(p_ids) > 0
        )
        coverage_rate = (
            round(subqueries_with_evidence / total_subqueries, 4)
            if total_subqueries > 0
            else 1.0
        )

        unique_sources = len({p.source_id or p.url for p in merged_passages if (p.source_id or p.url)})
        unique_documents = len({p.document_id for p in merged_passages if p.document_id})


        coverage_metrics = EvidenceCoverageMetrics(
            subqueries=total_subqueries,
            subqueries_with_evidence=subqueries_with_evidence,
            coverage=coverage_rate,
            unique_sources=unique_sources,
            unique_documents=unique_documents,
            failed_subqueries=sorted(list(failed_ids)),
        )

        merge_lat = round((time.perf_counter() - start_time) * 1000, 2)

        planning_metadata = {
            "strategy": plan.strategy,
            "subquery_count": total_subqueries,
            "subqueries": [sq.model_dump() for sq in plan.subqueries],
            "planning_latency_ms": planning_latency_ms,
            "merge_latency_ms": merge_lat,
        }

        # Build aggregated provenance
        aggregated_provenance: dict[str, Any] = {
            "subqueries": [sq.model_dump() for sq in plan.subqueries],
            "subquery_evidence_map": subquery_evidence_map,
            "coverage_metrics": coverage_metrics.model_dump(),
        }

        return MergedEvidencePack(
            query=plan.original_query,
            selected_passages=merged_passages,
            total_candidates=total_candidate_count,
            total_evidence=len(merged_passages),
            retrieval_metadata=all_retrieval_meta,
            ranking_metadata=all_ranking_meta,
            provenance=aggregated_provenance,
            subquery_evidence_map=subquery_evidence_map,
            coverage_metrics=coverage_metrics,
            planning_metadata=planning_metadata,
        )

"""
Strongly typed data models for Query Intelligence, Intent Classification,
Complexity Detection, Research Planning, and Evidence Merging.
"""
from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from ranking.evidence import EvidencePack, EvidencePassage


class QueryIntent(str, Enum):
    """Semantic intent classification categories for research queries."""
    FACTUAL = "FACTUAL"
    COMPARATIVE = "COMPARATIVE"
    CAUSAL = "CAUSAL"
    EXPLANATORY = "EXPLANATORY"
    PROCEDURAL = "PROCEDURAL"
    MULTI_HOP = "MULTI_HOP"
    SURVEY = "SURVEY"
    TREND = "TREND"
    UNKNOWN = "UNKNOWN"


class QueryComplexity(str, Enum):
    """Estimated structural complexity of the research query."""
    SIMPLE = "SIMPLE"
    MODERATE = "MODERATE"
    COMPLEX = "COMPLEX"


class QueryAnalysis(BaseModel):
    """
    Structured query understanding output capturing intent, complexity,
    entities, concepts, and constraints.
    """
    original_query: str = Field(..., description="Original user research query")
    intent: QueryIntent = Field(default=QueryIntent.FACTUAL, description="Classified query intent")
    complexity: QueryComplexity = Field(default=QueryComplexity.SIMPLE, description="Query structural complexity")
    requires_decomposition: bool = Field(
        default=False,
        description="Whether query warrants decomposition into multiple sub-queries",
    )
    entities: list[str] = Field(default_factory=list, description="Extracted key named entities or technical terms")
    concepts: list[str] = Field(default_factory=list, description="Key research topics or thematic concepts")
    constraints: list[str] = Field(default_factory=list, description="Explicit search constraints (e.g. methods, datasets)")
    time_constraints: list[str] = Field(default_factory=list, description="Temporal constraints (e.g. 'recent', '2024')")
    source_requirements: list[str] = Field(default_factory=list, description="Requested source types (e.g. 'arxiv', 'code')")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score of the analysis")


class ResearchSubQuery(BaseModel):
    """
    A single focused, independently searchable research sub-query.
    """
    subquery_id: str = Field(..., description="Unique subquery identifier (e.g. 'sq_001')")
    text: str = Field(..., min_length=1, description="Targeted search query text formulated for retrieval")
    purpose: str = Field(..., description="Information need addressed by this sub-query")
    intent: QueryIntent = Field(default=QueryIntent.FACTUAL, description="Intent specific to this subquery")
    priority: int = Field(default=1, ge=1, le=5, description="Execution priority (1 = highest)")
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of prerequisite subqueries that provide context",
    )


class ResearchPlan(BaseModel):
    """
    Complete research strategy decomposing the user question into subqueries.
    """
    original_query: str = Field(..., description="Original user question")
    analysis: QueryAnalysis = Field(..., description="Pre-decomposition query analysis")
    subqueries: list[ResearchSubQuery] = Field(
        default_factory=list,
        description="Ordered set of targeted research sub-queries",
    )
    strategy: str = Field(
        default="parallel_independent",
        description="Execution strategy (e.g. 'single_pass', 'parallel_independent', 'dependency_aware')",
    )
    execution_order: list[list[str]] = Field(
        default_factory=list,
        description="Batches of subquery IDs that can execute concurrently",
    )


class EvidenceCoverageMetrics(BaseModel):
    """
    Deterministic measurements of evidence coverage and source diversity across subqueries.
    """
    subqueries: int = Field(default=0, description="Total number of planned subqueries")
    subqueries_with_evidence: int = Field(default=0, description="Number of subqueries that yielded evidence")
    coverage: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of subqueries with evidence (subqueries_with_evidence / subqueries)")
    unique_sources: int = Field(default=0, description="Count of distinct canonical source IDs or URLs")
    unique_documents: int = Field(default=0, description="Count of distinct parent document IDs")
    failed_subqueries: list[str] = Field(default_factory=list, description="Subqueries that failed retrieval")


class MergedEvidencePack(EvidencePack):
    """
    Unified EvidencePack extending standard EvidencePack with subquery provenance,
    coverage metrics, and execution metadata.
    """
    subquery_evidence_map: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Mapping from subquery_id to list of selected passage_ids",
    )
    coverage_metrics: EvidenceCoverageMetrics = Field(
        default_factory=EvidenceCoverageMetrics,
        description="Deterministic evidence coverage and diversity metrics",
    )
    planning_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata from query analysis and decomposition stage",
    )

"""
Query Intelligence and Research Decomposition package.
Provides query analysis, intent classification, complexity detection,
subquery decomposition, parallel retrieval orchestration, and evidence merging.
"""
from __future__ import annotations

from query.complexity import QueryComplexityAnalyzer
from query.decomposer import QueryDecomposer
from query.intent import QueryIntentClassifier
from query.merger import EvidenceMerger
from query.models import (
    EvidenceCoverageMetrics,
    MergedEvidencePack,
    QueryAnalysis,
    QueryComplexity,
    QueryIntent,
    ResearchPlan,
    ResearchSubQuery,
)
from query.planner import (
    PlanningLatencyProfile,
    ResearchPlanRetrievalRequest,
    ResearchPlanningService,
)

__all__ = [
    "EvidenceCoverageMetrics",
    "EvidenceMerger",
    "MergedEvidencePack",
    "PlanningLatencyProfile",
    "QueryAnalysis",
    "QueryComplexity",
    "QueryComplexityAnalyzer",
    "QueryDecomposer",
    "QueryIntent",
    "QueryIntentClassifier",
    "ResearchPlan",
    "ResearchPlanRetrievalRequest",
    "ResearchPlanningService",
    "ResearchSubQuery",
]

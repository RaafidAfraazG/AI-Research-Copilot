"""
Deterministic Query Complexity Analyzer.
Classifies research questions into SIMPLE, MODERATE, or COMPLEX using structural,
lexical, and semantic heuristics without mandatory LLM dependencies.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from query.models import QueryComplexity


class ComplexitySignals(NamedTuple):
    """Signals extracted from the query for complexity scoring."""
    token_count: int
    clause_count: int
    conjunction_count: int
    has_comparative: bool
    has_causal: bool
    has_multi_request: bool
    has_temporal: bool
    entity_count: int
    question_mark_count: int
    score: float


class QueryComplexityAnalyzer:
    """
    Deterministic analyzer identifying whether a query requires decomposition.
    """

    # Comparative keywords and patterns
    COMPARATIVE_PATTERNS = [
        r"\b(?:compare|comparison|versus|vs\.?|differ(?:ence|s|ing)?|distinguish|trade-?offs?)\b",
        r"\b(?:pros\s+and\s+cons|strengths\s+and\s+weaknesses|advantages\s+and\s+(?:disadvantages|limitations))\b",
        r"\b(?:better\s+than|worse\s+than|superior\s+to|inferior\s+to|relative\s+to)\b",
        r"\b(?:when\s+to\s+use\s+\w+\s+(?:over|versus|vs|instead\s+of))\b",
    ]

    # Multi-request indicators
    MULTI_REQUEST_PATTERNS = [
        r"\b(?:and\s+also|as\s+well\s+as|in\s+addition(?:\s+to)?|along\s+with|furthermore|moreover)\b",
        r"\b(?:explain\s+.+\s+and\s+(?:summarize|evaluate|compare|identify|analyze))\b",
        r"\b(?:how\s+(?:does|do|can)\s+.+\s+and\s+(?:what|why|how))\b",
        r"\b(?:what\s+are\s+.+\s+and\s+(?:how|why|what))\b",
        r"\b(?:strengths?|benefits?|advantages?)\s+(?:and|as\s+well\s+as)\s+(?:weaknesses?|drawbacks?|limitations?)\b",
        r"\b(?:how\s+to\s+.+\s+and\s+what\s+are)\b",
    ]

    # Causal / Explanatory indicators
    CAUSAL_PATTERNS = [
        r"\b(?:why\s+(?:does|do|is|are|would)|what\s+causes|how\s+does\s+.+\s+(?:affect|influence|lead\s+to|impact|struggle|fail))\b",
        r"\b(?:relationship\s+between|causal\s+effect|impact\s+of\s+.+\s+on)\b",
    ]

    # Temporal / Trend constraints
    TEMPORAL_PATTERNS = [
        r"\b(?:recent(?:ly)?|latest|trends?|state-?of-?the-?art|sota|future\s+directions?|historical\s+evolution|advancements?)\b",
        r"\b(?:202[0-9]|201[5-9])\b",
    ]

    # Multi-hop patterns
    MULTI_HOP_PATTERNS = [
        r"\b(?:which\s+.+\s+(?:used|proposed|evaluated)\s+in\s+the\s+paper\s+that)\b",
        r"\b(?:benchmarks?\s+(?:were\s+)?used\s+in\s+.+\s+and\s+how\s+did)\b",
        r"\b(?:find\s+papers?\s+(?:citing|by\s+the\s+author\s+of))\b",
    ]

    # Conjunctions that join clauses
    CONJUNCTION_PATTERN = re.compile(
        r"\b(?:and|or|while|whereas|although|though|however|plus|but)\b",
        re.IGNORECASE,
    )

    # Common technical terms for entity counting
    TECHNICAL_TERM_PATTERN = re.compile(
        r"\b(?:rag|bm25|dense\s+retrieval|sparse\s+retrieval|cross-?encoders?|bi-?encoders?|llms?|transformers?|"
        r"vector\s+(?:search|database|index)|hybrid\s+retrieval|reranking|fine-?tuning|embeddings?|"
        r"prompt\s+engineering|hallucination|grounding|evaluation|knowledge\s+graphs?|qdrant|opensearch|"
        r"colbert|splade|dpr|dense\s+passage\s+retrieval|bge|sentence-?transformers?|attention\s+mechanism)\b",
        re.IGNORECASE,
    )

    @classmethod
    def extract_signals(cls, query: str) -> ComplexitySignals:
        """Extract lexical, structural, and semantic signals from the query."""
        clean_text = query.strip()
        tokens = [t for t in re.split(r"\s+", clean_text) if t]
        token_count = len(tokens)

        # Clauses: split on semicolons, commas followed by conjunctions, or '?'
        clause_splits = re.split(r"[;?]|,\s*(?:and|or|while|whereas|as well as|but)", clean_text)
        clause_count = max(1, len([c for c in clause_splits if c.strip()]))

        # Conjunctions
        conjunctions = cls.CONJUNCTION_PATTERN.findall(clean_text)
        conjunction_count = len(conjunctions)

        # Pattern matches
        has_comparative = any(bool(re.search(p, clean_text, re.IGNORECASE)) for p in cls.COMPARATIVE_PATTERNS)
        has_multi_request = any(bool(re.search(p, clean_text, re.IGNORECASE)) for p in cls.MULTI_REQUEST_PATTERNS)
        has_causal = any(bool(re.search(p, clean_text, re.IGNORECASE)) for p in cls.CAUSAL_PATTERNS)
        has_temporal = any(bool(re.search(p, clean_text, re.IGNORECASE)) for p in cls.TEMPORAL_PATTERNS)
        has_multi_hop = any(bool(re.search(p, clean_text, re.IGNORECASE)) for p in cls.MULTI_HOP_PATTERNS)

        # Entity count
        entities = cls.TECHNICAL_TERM_PATTERN.findall(clean_text)
        cap_entities = re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b", clean_text)
        unique_entities = set(e.lower() for e in entities) | set(e.lower() for e in cap_entities)
        # Filter common words from entities
        unique_entities = {e for e in unique_entities if e not in {"what", "how", "why", "which", "compare", "define", "the"}}
        entity_count = len(unique_entities)

        question_mark_count = clean_text.count("?")

        # Calculate score in [0.0, 10.0]
        score = 0.0
        if token_count > 20:
            score += 3.5
        elif token_count > 12:
            score += 2.0
        elif token_count >= 9:
            score += 1.5

        if clause_count >= 3:
            score += 2.5
        elif clause_count == 2:
            score += 1.5

        if conjunction_count >= 3:
            score += 2.5
        elif conjunction_count >= 1:
            score += 1.0

        if has_comparative:
            score += 2.5
        if has_multi_request:
            score += 2.5
        if has_causal:
            score += 1.5
        if has_temporal:
            score += 1.5
        if has_multi_hop:
            score += 2.5


        if re.search(r"\bhow\s+to\b", clean_text, re.IGNORECASE) and token_count >= 9:
            score += 1.5
        if re.search(r"\bhow\s+does\b", clean_text, re.IGNORECASE) and token_count >= 9:
            score += 1.0

        if entity_count >= 3:
            score += 2.5
        elif entity_count == 2:
            score += 1.5
        elif entity_count == 1 and token_count >= 9:
            score += 0.5

        if question_mark_count >= 2:
            score += 2.0


        return ComplexitySignals(
            token_count=token_count,
            clause_count=clause_count,
            conjunction_count=conjunction_count,
            has_comparative=has_comparative,
            has_causal=has_causal,
            has_multi_request=has_multi_request,
            has_temporal=has_temporal,
            entity_count=entity_count,
            question_mark_count=question_mark_count,
            score=round(score, 2),
        )

    @classmethod
    def analyze(cls, query: str) -> tuple[QueryComplexity, bool]:
        """
        Analyze query and return (QueryComplexity, requires_decomposition).

        Classification rules:
        - SIMPLE (< 2.5 score, short, single entity/topic): requires_decomposition = False
        - MODERATE (2.5 <= score < 5.5, 2 facets/conjunction): requires_decomposition = True
        - COMPLEX (score >= 5.5, multi-faceted/comparative/temporal/multi-hop): requires_decomposition = True
        """
        clean_text = query.strip()
        if not clean_text:
            return QueryComplexity.SIMPLE, False

        signals = cls.extract_signals(clean_text)

        # Direct shortcuts for canonical simple queries:
        is_short_single = (
            signals.token_count <= 8
            and signals.clause_count <= 1
            and not signals.has_comparative
            and not signals.has_multi_request
            and signals.conjunction_count == 0
            and signals.question_mark_count <= 1
            and signals.entity_count <= 1
        )

        if is_short_single:
            return QueryComplexity.SIMPLE, False

        if signals.score < 2.5:
            return QueryComplexity.SIMPLE, False
        elif signals.score < 5.5:
            return QueryComplexity.MODERATE, True
        else:
            return QueryComplexity.COMPLEX, True

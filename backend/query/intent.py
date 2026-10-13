"""
Layered Query Intent Classifier.
Implements Level 1 deterministic rule-based pattern classification with an
optional Level 2 LLM classifier for ambiguous research queries.
"""
from __future__ import annotations

import json
import re
import structlog

from llm.base import LLMProvider
from query.models import QueryIntent

logger = structlog.get_logger(__name__)


class QueryIntentClassifier:
    """
    Two-stage intent classifier:
    1. Deterministic heuristic rules (fast, offline, reproducible)
    2. LLM classification fallback for ambiguous queries
    """

    COMPARATIVE_PATTERNS = [
        r"\b(?:compare|comparison|versus|vs\.?|differ(?:ence|s|ing)?|distinguish|trade-?offs?)\b",
        r"\b(?:pros\s+and\s+cons|strengths\s+and\s+weaknesses|advantages\s+and\s+(?:disadvantages|limitations))\b",
        r"\b(?:better\s+than|worse\s+than|superior\s+to|relative\s+to)\b",
        r"\b(?:which\s+(?:one\s+)?is\s+(?:better|faster|more\s+accurate|preferred))\b",
    ]

    TREND_PATTERNS = [
        r"\b(?:recent(?:ly)?|latest|trends?|state-?of-?the-?art|sota|future\s+directions?|advancements?|emerging)\b",
        r"\b(?:in\s+202[3-6]|since\s+202[0-9]|newest\s+developments?)\b",
    ]

    SURVEY_PATTERNS = [
        r"\b(?:survey|overview|taxonomy|landscape|literature\s+review|comprehensive\s+review|broad\s+perspective)\b",
        r"\b(?:summarize\s+all|summary\s+of\s+methods|categories\s+of|types\s+of)\b",
    ]

    PROCEDURAL_PATTERNS = [
        r"\b(?:how\s+to\s+(?:implement|build|train|fine-?tune|deploy|configure|optimize|set\s+up|design))\b",
        r"\b(?:step-?by-?step|steps\s+to|guide\s+for|tutorial|workflow|pipeline\s+setup)\b",
        r"\b(?:code\s+example|implementation\s+details)\b",
    ]

    MULTI_HOP_PATTERNS = [
        r"\b(?:which\s+.+\s+(?:used|proposed|evaluated)\s+in\s+the\s+paper\s+that)\b",
        r"\b(?:find\s+papers?\s+(?:citing|by\s+the\s+author\s+of)\s+.+\s+and\s+also)\b",
        r"\b(?:papers?\s+that\s+combine\s+.+\s+and\s+.+\s+for)\b",
        r"\b(?:benchmarks?\s+(?:were\s+)?used\s+in\s+.+\s+and\s+how\s+did)\b",
    ]

    CAUSAL_PATTERNS = [
        r"\b(?:why\s+(?:does|do|is|are|would)|what\s+causes|cause\s+of|leading\s+to)\b",
        r"\b(?:reason\s+for|root\s+cause|why\s+should|why\s+would)\b",
        r"\b(?:how\s+does\s+.+\s+(?:affect|influence|impact|improve|degrade|reduce))\b",
        r"\b(?:why\s+.+\s+(?:struggle|fail|hallucinate|degrade))\b",
    ]

    EXPLANATORY_PATTERNS = [
        r"\b(?:how\s+(?:does|do|can)\s+.+\s+work)\b",
        r"\b(?:explain|explanation|how\s+is\s+.+\s+calculated|mechanism\s+behind|inner\s+workings)\b",
        r"\b(?:how\s+does\s+.+\s+function|how\s+it\s+works)\b",
        r"\b(?:how\s+does\s+.+\s+improve)\b",
    ]

    FACTUAL_PATTERNS = [
        r"^(?:what\s+is|what\s+are|define|definition\s+of|who\s+introduced|when\s+was|where\s+is)\b",
        r"\b(?:name\s+of|formula\s+for|meaning\s+of)\b",
    ]

    @classmethod
    def classify_deterministic(cls, query: str) -> tuple[QueryIntent, float]:
        """
        Deterministic Level 1 intent classification.
        Returns (QueryIntent, confidence).
        """
        clean_query = query.strip()
        if not clean_query:
            return QueryIntent.UNKNOWN, 0.0

        # Multi-hop detection takes precedence for complex linked research queries
        if any(re.search(p, clean_query, re.IGNORECASE) for p in cls.MULTI_HOP_PATTERNS):
            return QueryIntent.MULTI_HOP, 0.95

        # Procedural (How to design, how to fine-tune, etc.)
        if any(re.search(p, clean_query, re.IGNORECASE) for p in cls.PROCEDURAL_PATTERNS):
            return QueryIntent.PROCEDURAL, 0.95

        # Comparative (Compare X and Y, trade-offs between, etc.)
        if any(re.search(p, clean_query, re.IGNORECASE) for p in cls.COMPARATIVE_PATTERNS):
            return QueryIntent.COMPARATIVE, 0.95

        # Trend (Recent trends, 2024, 2025, etc.)
        if any(re.search(p, clean_query, re.IGNORECASE) for p in cls.TREND_PATTERNS):
            return QueryIntent.TREND, 0.90

        # Survey (Comprehensive survey, landscape, etc.)
        if any(re.search(p, clean_query, re.IGNORECASE) for p in cls.SURVEY_PATTERNS):
            return QueryIntent.SURVEY, 0.90

        # Explanatory (How does ... work, explain ..., how does ... improve)
        if any(re.search(p, clean_query, re.IGNORECASE) for p in cls.EXPLANATORY_PATTERNS):
            return QueryIntent.EXPLANATORY, 0.90

        # Causal (Why does ..., what causes ...)
        if any(re.search(p, clean_query, re.IGNORECASE) for p in cls.CAUSAL_PATTERNS):
            return QueryIntent.CAUSAL, 0.90


        # Factual (What is ..., Define ...)
        if any(re.search(p, clean_query, re.IGNORECASE) for p in cls.FACTUAL_PATTERNS):
            return QueryIntent.FACTUAL, 0.90

        # General "what" / "which" questions default to FACTUAL
        if re.search(r"^(?:what|which)\b", clean_query, re.IGNORECASE):
            return QueryIntent.FACTUAL, 0.75

        # Fallback to EXPLANATORY if it starts with "how"
        if re.search(r"^how\b", clean_query, re.IGNORECASE):
            return QueryIntent.EXPLANATORY, 0.70

        return QueryIntent.FACTUAL, 0.50

    @classmethod
    async def classify(
        cls,
        query: str,
        llm_provider: LLMProvider | None = None,
        use_llm: bool = False,
    ) -> tuple[QueryIntent, float]:
        """
        Classify intent using deterministic rules first, falling back to LLM if ambiguous.
        """
        det_intent, confidence = cls.classify_deterministic(query)

        if not use_llm or llm_provider is None or confidence >= 0.80:
            return det_intent, confidence

        prompt = (
            f"Classify the research intent of this user question into exactly ONE of the following categories:\n"
            f"- FACTUAL: Direct definition or fact\n"
            f"- COMPARATIVE: Comparing two or more methods, systems, or tradeoffs\n"
            f"- CAUSAL: Explaining why something happens or causal relationships\n"
            f"- EXPLANATORY: Explaining how a mechanism or system works\n"
            f"- PROCEDURAL: Step-by-step instructions or implementation methods\n"
            f"- MULTI_HOP: Multi-step inference across linked literature\n"
            f"- SURVEY: Broad overview or landscape review\n"
            f"- TREND: Recent advancements or modern temporal state-of-the-art\n\n"
            f"User Question: \"{query}\"\n\n"
            f"Respond with JSON: {{\"intent\": \"CATEGORY_NAME\", \"confidence\": 0.95}}"
        )

        try:
            resp = await llm_provider.generate(
                prompt=prompt,
                system_prompt="You are a precise scientific NLP query classifier. Respond ONLY with valid JSON.",
                temperature=0.0,
                response_format="json",
            )
            raw_text = resp.content.strip()
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
            if match:
                raw_text = match.group(1).strip()
            data = json.loads(raw_text)
            intent_str = data.get("intent", "").upper().strip()
            if intent_str in QueryIntent.__members__:
                llm_conf = float(data.get("confidence", 0.90))
                return QueryIntent[intent_str], llm_conf
        except Exception as e:
            logger.warning("llm_intent_classification_failed", error=str(e), fallback=det_intent.value)

        return det_intent, confidence

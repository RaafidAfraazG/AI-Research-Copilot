"""
Query Decomposition Engine.
Decomposes complex research questions into targeted, searchable sub-queries
with deterministic rule-based formulation and LLM-assisted decomposition.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal
import structlog

from llm.base import LLMProvider
from query.complexity import QueryComplexityAnalyzer
from query.intent import QueryIntentClassifier
from query.models import (
    QueryAnalysis,
    QueryComplexity,
    QueryIntent,
    ResearchPlan,
    ResearchSubQuery,
)

logger = structlog.get_logger(__name__)


class QueryDecomposer:
    """
    Decomposes research questions into structured ResearchPlans.
    Supports both rule-based deterministic decomposition and LLM-assisted decomposition.
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        max_subqueries: int = 5,
    ) -> None:
        self.llm_provider = llm_provider
        self.max_subqueries = max_subqueries

    # ── Query Analysis ────────────────────────────────────────────────────────

    @classmethod
    def extract_entities_and_constraints(cls, query: str) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
        """
        Extract entities, concepts, general constraints, temporal constraints, and source requirements.
        """
        clean_text = query.strip()

        # Technical Entities & Acronyms
        entity_patterns = [
            r"\b(?:rag|bm25|dense\s+retrieval|sparse\s+retrieval|cross-?encoders?|bi-?encoders?|colbert|splade|dpr|bge|qdrant|opensearch|transformers?|llms?|sentence-?transformers?)\b",
            r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b",
        ]
        entities_found: set[str] = set()
        for p in entity_patterns:
            matches = re.findall(p, clean_text, re.IGNORECASE)
            for m in matches:
                if len(m.strip()) > 1 and m.lower() not in {"what", "how", "why", "which", "compare", "explain", "the", "and", "for"}:
                    entities_found.add(m.strip())

        # Concepts
        concept_patterns = [
            r"\b(?:retrieval\s+augmented\s+generation|hybrid\s+retrieval|reciprocal\s+rank\s+fusion|reranking|hallucination|grounding|vector\s+search|chunking|embeddings?|semantic\s+search)\b",
            r"\b(?:strengths\s+and\s+weaknesses|trade-?offs?|limitations?|failure\s+modes?|evaluation\s+metrics?)\b",
        ]
        concepts_found: set[str] = set()
        for p in concept_patterns:
            matches = re.findall(p, clean_text, re.IGNORECASE)
            concepts_found.update(m.strip() for m in matches)

        # Temporal constraints
        time_patterns = [
            r"\b(?:recent(?:ly)?|latest|trends?|state-?of-?the-?art|sota|future\s+directions?|202[0-9]|201[5-9])\b"
        ]
        time_constraints: set[str] = set()
        for p in time_patterns:
            matches = re.findall(p, clean_text, re.IGNORECASE)
            time_constraints.update(m.strip() for m in matches)

        # General constraints
        constraints_found: set[str] = set()
        if re.search(r"\b(?:strengths|advantages|benefits)\b", clean_text, re.IGNORECASE):
            constraints_found.add("strengths")
        if re.search(r"\b(?:weaknesses|drawbacks|limitations)\b", clean_text, re.IGNORECASE):
            constraints_found.add("limitations")
        if re.search(r"\b(?:when\s+(?:each\s+)?performs\s+better|preferable|best\s+scenarios)\b", clean_text, re.IGNORECASE):
            constraints_found.add("decision_criteria")

        # Source requirements
        source_requirements: list[str] = []
        if re.search(r"\b(?:arxiv|papers?|academic|literature|research)\b", clean_text, re.IGNORECASE):
            source_requirements.append("arxiv")
        if re.search(r"\b(?:code|github|implementation|repository)\b", clean_text, re.IGNORECASE):
            source_requirements.append("github")
        if re.search(r"\b(?:web|docs|documentation|article)\b", clean_text, re.IGNORECASE):
            source_requirements.append("web")

        return (
            sorted(list(entities_found)),
            sorted(list(concepts_found)),
            sorted(list(constraints_found)),
            sorted(list(time_constraints)),
            sorted(list(set(source_requirements))),
        )

    @classmethod
    def analyze_query(
        cls,
        query: str,
        intent_override: QueryIntent | None = None,
    ) -> QueryAnalysis:
        """Analyze query complexity, intent, entities, and constraints."""
        clean_text = query.strip()
        complexity, req_decomp = QueryComplexityAnalyzer.analyze(clean_text)
        intent, conf = (
            (intent_override, 1.0)
            if intent_override
            else QueryIntentClassifier.classify_deterministic(clean_text)
        )
        entities, concepts, constraints, time_c, sources = cls.extract_entities_and_constraints(clean_text)

        return QueryAnalysis(
            original_query=clean_text,
            intent=intent,
            complexity=complexity,
            requires_decomposition=req_decomp,
            entities=entities,
            concepts=concepts,
            constraints=constraints,
            time_constraints=time_c,
            source_requirements=sources,
            confidence=conf,
        )

    # ── Deterministic Rule-Based Decomposition ────────────────────────────────

    def decompose_deterministic(self, query: str, analysis: QueryAnalysis) -> ResearchPlan:
        """
        Decompose a research question using deterministic rules.
        Guarantees fast, offline, reproducible, and constraint-preserving plans.
        """
        clean_text = query.strip()

        # If decomposition is not required, return single subquery
        if not analysis.requires_decomposition:
            sub = ResearchSubQuery(
                subquery_id="sq_001",
                text=clean_text,
                purpose="Retrieve direct evidence to answer the research question",
                intent=analysis.intent,
                priority=1,
                dependencies=[],
            )
            return ResearchPlan(
                original_query=clean_text,
                analysis=analysis,
                subqueries=[sub],
                strategy="single_pass",
                execution_order=[["sq_001"]],
            )

        subqueries: list[ResearchSubQuery] = []

        # ── Category 1: Comparative Queries ──────────────────────────────────
        if analysis.intent == QueryIntent.COMPARATIVE:
            # Look for comparison targets (e.g., "X and Y", "X vs Y", "between X and Y")
            comp_match = re.search(
                r"(?:compare|between|vs\.?|versus)\s+([^,]+?)\s+(?:and|versus|vs\.?|with)\s+([^,]+?)(?:(?=,\s*(?:explain|identify|summarize)|(?:\s+and\s+(?:explain|identify|summarize))|\s+in\s+terms\s+of|\s+for\s+search|\s+for\s+RAG|\?|$))",
                clean_text,
                re.IGNORECASE,
            )

            target_a = "Method A"
            target_b = "Method B"
            domain_context = ""

            if "rag" in clean_text.lower():
                domain_context = " in RAG systems"

            if comp_match:
                target_a = comp_match.group(1).strip()
                target_b = comp_match.group(2).strip()
            elif len(analysis.entities) >= 2:
                target_a = analysis.entities[0]
                target_b = analysis.entities[1]

            # 1. Target A overview & strengths
            subqueries.append(
                ResearchSubQuery(
                    subquery_id="sq_001",
                    text=f"{target_a}{domain_context} architectures mechanisms and retrieval strengths",
                    purpose=f"Understand {target_a} architecture, mechanisms, and advantages{domain_context}",
                    intent=QueryIntent.EXPLANATORY,
                    priority=1,
                    dependencies=[],
                )
            )

            # 2. Target B overview & strengths
            subqueries.append(
                ResearchSubQuery(
                    subquery_id="sq_002",
                    text=f"{target_b}{domain_context} architectures mechanisms and retrieval strengths",
                    purpose=f"Understand {target_b} architecture, mechanisms, and advantages{domain_context}",
                    intent=QueryIntent.EXPLANATORY,
                    priority=1,
                    dependencies=[],
                )
            )

            # Check for specific comparative dimensions (e.g. latency, accuracy, memory, trade-offs)
            dim_matches = re.findall(r"\b(?:latency|accuracy|memory\s+overhead|throughput|computational\s+cost|trade-?offs?)\b", clean_text, re.IGNORECASE)
            dim_str = f" {' '.join(dim_matches)}" if dim_matches else " trade-offs benchmark performance"

            # 3. Direct comparison & tradeoffs
            subqueries.append(
                ResearchSubQuery(
                    subquery_id="sq_003",
                    text=f"{target_a} vs {target_b}{domain_context} comparative evaluation{dim_str}",
                    purpose=f"Compare performance, trade-offs, and benchmarks between {target_a} and {target_b}",
                    intent=QueryIntent.COMPARATIVE,
                    priority=1,
                    dependencies=["sq_001", "sq_002"],
                )
            )

            # 4. Strengths, weaknesses, or selection criteria if requested
            if "limitations" in analysis.constraints or "strengths" in analysis.constraints or "decision_criteria" in analysis.constraints or re.search(r"when|strengths|weaknesses|perform|better", clean_text, re.IGNORECASE):
                subqueries.append(
                    ResearchSubQuery(
                        subquery_id="sq_004",
                        text=f"when to use {target_a} vs {target_b}{domain_context} failure cases limitations best practices",
                        purpose=f"Identify selection criteria, failure modes, and optimal use cases for {target_a} and {target_b}",
                        intent=QueryIntent.COMPARATIVE,
                        priority=2,
                        dependencies=["sq_003"],
                    )
                )

            # 5. Temporal / Trends if requested
            if analysis.time_constraints or any("trend" in t.lower() or "recent" in t.lower() for t in analysis.time_constraints):
                subqueries.append(
                    ResearchSubQuery(
                        subquery_id=f"sq_{len(subqueries)+1:03d}",
                        text=f"recent advances and research trends in {target_a} and {target_b}{domain_context} 2024",
                        purpose=f"Summarize recent research trends and latest developments in {target_a} and {target_b}",
                        intent=QueryIntent.TREND,
                        priority=2,
                        dependencies=["sq_001", "sq_002"],
                    )
                )


        # ── Category 2: Multi-Part Conjunction Queries ───────────────────────
        elif "and" in clean_text.lower() or ";" in clean_text or "?" in clean_text:
            parts = re.split(r";|\?|,\s*(?:and|as well as)|\s+and\s+(?:what|how|why|which|explain|summarize)", clean_text)
            parts = [p.strip() for p in parts if len(p.strip()) > 5]

            if len(parts) >= 2:
                for idx, part in enumerate(parts[:self.max_subqueries], start=1):
                    # Clean search query text
                    clean_part = re.sub(r"^(?:how\s+does|what\s+are|explain|summarize|identify|tell\s+me)\s+", "", part, flags=re.IGNORECASE).strip()
                    part_intent, _ = QueryIntentClassifier.classify_deterministic(part)
                    subqueries.append(
                        ResearchSubQuery(
                            subquery_id=f"sq_{idx:03d}",
                            text=f"{clean_part}",
                            purpose=f"Investigate: {part}",
                            intent=part_intent,
                            priority=1,
                            dependencies=[],
                        )
                    )
            else:
                # Fallback multi-part decomposition
                subqueries = self._default_multi_facet_decomposition(clean_text, analysis)

        # ── Category 3: Fallback General Decomposition ───────────────────────
        else:
            subqueries = self._default_multi_facet_decomposition(clean_text, analysis)

        # Enforce max subquery limit
        subqueries = subqueries[:self.max_subqueries]

        # Ensure unique IDs and normalize priorities
        subqueries = self._normalize_subqueries(subqueries)

        # Compute execution order batches based on dependencies
        execution_order = self._compute_execution_order(subqueries)

        return ResearchPlan(
            original_query=clean_text,
            analysis=analysis,
            subqueries=subqueries,
            strategy="parallel_independent" if len(execution_order) <= 1 else "dependency_aware",
            execution_order=execution_order,
        )

    def _default_multi_facet_decomposition(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> list[ResearchSubQuery]:
        """Generate focused multi-facet subqueries for complex questions."""
        subqueries: list[ResearchSubQuery] = []
        topic = query.strip()
        # Strip leading question words
        topic_core = re.sub(r"^(?:what\s+is|what\s+are|how\s+does|why\s+does|why\s+do|explain)\s+", "", topic, flags=re.IGNORECASE).rstrip("?")

        if analysis.intent == QueryIntent.CAUSAL:
            # 1. Root causes and mechanisms
            subqueries.append(
                ResearchSubQuery(
                    subquery_id="sq_001",
                    text=f"{topic_core} root causes and underlying mechanisms",
                    purpose=f"Investigate why {topic_core} occurs and underlying causal mechanisms",
                    intent=QueryIntent.CAUSAL,
                    priority=1,
                    dependencies=[],
                )
            )
            # 2. Impact on systems
            subqueries.append(
                ResearchSubQuery(
                    subquery_id="sq_002",
                    text=f"{topic_core} impact on retrieval accuracy and hallucination rates",
                    purpose=f"Evaluate empirical effects and impact of {topic_core}",
                    intent=QueryIntent.CAUSAL,
                    priority=1,
                    dependencies=["sq_001"],
                )
            )
            # 3. Mitigation strategies
            subqueries.append(
                ResearchSubQuery(
                    subquery_id="sq_003",
                    text=f"{topic_core} mitigation methods architectural solutions",
                    purpose=f"Identify solutions, defenses, and mitigations for {topic_core}",
                    intent=QueryIntent.EXPLANATORY,
                    priority=2,
                    dependencies=["sq_001"],
                )
            )
        elif analysis.intent in (QueryIntent.TREND, QueryIntent.SURVEY):
            # 1. Core taxonomy
            subqueries.append(
                ResearchSubQuery(
                    subquery_id="sq_001",
                    text=f"{topic_core} foundational architectures and taxonomies",
                    purpose=f"Survey foundational concepts and architectural taxonomies for {topic_core}",
                    intent=QueryIntent.SURVEY,
                    priority=1,
                    dependencies=[],
                )
            )
            # 2. Recent advances
            time_tag = " 2024" if not any("202" in t for t in analysis.time_constraints) else ""
            subqueries.append(
                ResearchSubQuery(
                    subquery_id="sq_002",
                    text=f"{topic_core} state of the art advancements recent trends{time_tag}",
                    purpose=f"Identify latest advancements and state of the art techniques in {topic_core}",
                    intent=QueryIntent.TREND,
                    priority=1,
                    dependencies=["sq_001"],
                )
            )
            # 3. Open challenges
            subqueries.append(
                ResearchSubQuery(
                    subquery_id="sq_003",
                    text=f"{topic_core} open challenges future directions",
                    purpose=f"Analyze open research challenges and future directions for {topic_core}",
                    intent=QueryIntent.SURVEY,
                    priority=2,
                    dependencies=["sq_001"],
                )
            )
        else:
            # 1. Core mechanism / definition
            subqueries.append(
                ResearchSubQuery(
                    subquery_id="sq_001",
                    text=f"{topic_core} core architecture and mechanisms",
                    purpose=f"Investigate core concepts and architectural foundations of {topic_core}",
                    intent=QueryIntent.EXPLANATORY,
                    priority=1,
                    dependencies=[],
                )
            )

            # 2. Key benefits and applications
            subqueries.append(
                ResearchSubQuery(
                    subquery_id="sq_002",
                    text=f"{topic_core} benefits advantages and applications",
                    purpose=f"Identify primary advantages and practical applications of {topic_core}",
                    intent=QueryIntent.FACTUAL,
                    priority=1,
                    dependencies=[],
                )
            )

            # 3. Challenges / Limitations
            subqueries.append(
                ResearchSubQuery(
                    subquery_id="sq_003",
                    text=f"{topic_core} limitations challenges and failure modes",
                    purpose=f"Examine limitations, vulnerabilities, and open challenges for {topic_core}",
                    intent=QueryIntent.EXPLANATORY,
                    priority=2,
                    dependencies=["sq_001"],
                )
            )

        return subqueries


    # ── LLM-Assisted Decomposition ───────────────────────────────────────────

    async def decompose_llm(
        self,
        query: str,
        analysis: QueryAnalysis,
        llm_provider: LLMProvider,
    ) -> ResearchPlan:
        """
        Decompose a research question into subqueries using the LLM provider.
        Falls back to deterministic decomposition if the LLM output is invalid or fails.
        """
        prompt = (
            f"USER QUESTION:\n{query}\n\n"
            f"TASK:\n"
            f"Break this research question into the smallest useful set of independent research sub-questions.\n\n"
            f"Requirements:\n"
            f"- Preserve the original research intent and scope.\n"
            f"- Cover every requested dimension, comparison target, entity, and constraint.\n"
            f"- Avoid redundant or trivial sub-questions.\n"
            f"- Formulate each sub-question as a concise, keyword-rich search query suitable for BM25 and neural vector search.\n"
            f"- Assign priority (1 = critical, 2 = secondary).\n"
            f"- Maximum {self.max_subqueries} sub-questions.\n"
            f"- Return structured JSON matching this schema:\n\n"
            f'{{\n'
            f'  "subqueries": [\n'
            f'    {{\n'
            f'      "subquery_id": "sq_001",\n'
            f'      "text": "targeted search query text",\n'
            f'      "purpose": "clear explanation of information need",\n'
            f'      "intent": "FACTUAL | COMPARATIVE | CAUSAL | EXPLANATORY | PROCEDURAL | SURVEY | TREND",\n'
            f'      "priority": 1,\n'
            f'      "dependencies": []\n'
            f'    }}\n'
            f'  ]\n'
            f'}}\n'
        )

        try:
            resp = await llm_provider.generate(
                prompt=prompt,
                system_prompt="You are a senior research planning assistant. Respond strictly in valid JSON.",
                temperature=0.0,
                response_format="json",
            )
            raw_text = resp.content.strip()
            # Clean markdown wrappers if present
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
            if match:
                raw_text = match.group(1).strip()

            data = json.loads(raw_text)
            raw_subqueries = data.get("subqueries", [])
            parsed_subqueries: list[ResearchSubQuery] = []

            for idx, sq in enumerate(raw_subqueries, start=1):
                if isinstance(sq, dict) and sq.get("text"):
                    sq_id = sq.get("subquery_id") or f"sq_{idx:03d}"
                    text = sq.get("text", "").strip()
                    purpose = sq.get("purpose") or f"Address facet {idx}"
                    intent_str = sq.get("intent", "FACTUAL").upper().strip()
                    intent_val = (
                        QueryIntent[intent_str]
                        if intent_str in QueryIntent.__members__
                        else QueryIntent.FACTUAL
                    )
                    priority = int(sq.get("priority", 1))
                    deps = sq.get("dependencies", [])
                    if isinstance(deps, list):
                        deps = [str(d) for d in deps if d]
                    else:
                        deps = []

                    parsed_subqueries.append(
                        ResearchSubQuery(
                            subquery_id=sq_id,
                            text=text,
                            purpose=purpose,
                            intent=intent_val,
                            priority=min(max(priority, 1), 5),
                            dependencies=deps,
                        )
                    )

            if parsed_subqueries:
                parsed_subqueries = self._normalize_subqueries(parsed_subqueries[:self.max_subqueries])
                execution_order = self._compute_execution_order(parsed_subqueries)
                return ResearchPlan(
                    original_query=query,
                    analysis=analysis,
                    subqueries=parsed_subqueries,
                    strategy="parallel_independent" if len(execution_order) <= 1 else "dependency_aware",
                    execution_order=execution_order,
                )

        except Exception as e:
            logger.warning("llm_decomposition_failed", error=str(e), fallback="deterministic")

        # Fallback to deterministic decomposition
        return self.decompose_deterministic(query, analysis)

    # ── High-Level Decomposition Dispatcher ──────────────────────────────────

    async def decompose(
        self,
        query: str,
        analysis: QueryAnalysis | None = None,
        mode: Literal["auto", "llm", "deterministic"] = "auto",
    ) -> ResearchPlan:
        """
        Main entrypoint for decomposing a research query into a ResearchPlan.
        """
        clean_query = query.strip()
        current_analysis = analysis or self.analyze_query(clean_query)

        # Simple queries never undergo LLM decomposition to optimize latency
        if not current_analysis.requires_decomposition:
            return self.decompose_deterministic(clean_query, current_analysis)

        if mode == "deterministic":
            return self.decompose_deterministic(clean_query, current_analysis)

        if mode in ("auto", "llm") and self.llm_provider:
            # If using mock provider or real LLM
            return await self.decompose_llm(clean_query, current_analysis, self.llm_provider)

        return self.decompose_deterministic(clean_query, current_analysis)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalize_subqueries(self, subqueries: list[ResearchSubQuery]) -> list[ResearchSubQuery]:
        """Normalize IDs, eliminate near-duplicates, and ensure validity."""
        seen_texts: set[str] = set()
        unique_subs: list[ResearchSubQuery] = []

        for idx, sq in enumerate(subqueries, start=1):
            normalized_text = re.sub(r"\s+", " ", sq.text.lower().strip())
            # Deduplicate if exact or near-identical text
            if normalized_text in seen_texts:
                continue
            seen_texts.add(normalized_text)

            sq.subquery_id = f"sq_{len(unique_subs)+1:03d}"
            unique_subs.append(sq)

        return unique_subs

    def _compute_execution_order(self, subqueries: list[ResearchSubQuery]) -> list[list[str]]:
        """
        Compute staged execution order batches for parallel retrieval.
        Batch 1: Subqueries with no prerequisites.
        Batch 2+: Subqueries with fulfilled prerequisites.
        """
        if not subqueries:
            return []

        all_ids = {sq.subquery_id for sq in subqueries}
        independent = [sq.subquery_id for sq in subqueries if not sq.dependencies or not set(sq.dependencies).issubset(all_ids)]
        dependent = [sq.subquery_id for sq in subqueries if sq.subquery_id not in independent]

        batches: list[list[str]] = []
        if independent:
            batches.append(independent)
        if dependent:
            batches.append(dependent)

        return batches if batches else [[sq.subquery_id for sq in subqueries]]

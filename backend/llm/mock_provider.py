"""
Deterministic Mock LLM Provider for unit testing and offline development.
Produces structured, grounded research answers, decomposition plans, and intent classifications
without external network calls or API keys.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Literal
import structlog

from llm.base import LLMProvider
from llm.models import LLMResponse, LLMUsage

logger = structlog.get_logger(__name__)


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider returning deterministic research synthesis and query planning outputs.
    """

    def __init__(
        self,
        custom_response: str | None = None,
        model_name: str = "mock-research-model-v1",
    ) -> None:
        self.custom_response = custom_response
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> LLMResponse:
        """Generate deterministic mock outputs based on prompt context."""
        start_time = time.perf_counter()

        if self.custom_response:
            content = self.custom_response
        elif "Classify the research intent" in prompt:
            # Intent classification mock response
            intent_val = "FACTUAL"
            if re.search(r"compare|versus|vs|trade-?off", prompt, re.IGNORECASE):
                intent_val = "COMPARATIVE"
            elif re.search(r"trend|recent|latest|2024", prompt, re.IGNORECASE):
                intent_val = "TREND"
            elif re.search(r"why|cause", prompt, re.IGNORECASE):
                intent_val = "CAUSAL"
            elif re.search(r"how\s+to|step", prompt, re.IGNORECASE):
                intent_val = "PROCEDURAL"
            elif re.search(r"how\s+does|explain", prompt, re.IGNORECASE):
                intent_val = "EXPLANATORY"

            content = json.dumps({"intent": intent_val, "confidence": 0.95}, indent=2)
        elif "Break this research question into the smallest useful set" in prompt or "sub-questions" in prompt:
            # Decomposition mock response
            q_match = re.search(r"USER QUESTION:\s*\n(.*?)(?:\n\n|\nTASK:)", prompt, re.DOTALL)
            extracted_q = q_match.group(1).strip() if q_match else "research topic"

            mock_subs = [
                {
                    "subquery_id": "sq_001",
                    "text": f"{extracted_q} foundational concepts and architectures",
                    "purpose": "Investigate core architectural concepts and definitions",
                    "intent": "EXPLANATORY",
                    "priority": 1,
                    "dependencies": [],
                },
                {
                    "subquery_id": "sq_002",
                    "text": f"{extracted_q} comparative benchmark evaluations and trade-offs",
                    "purpose": "Analyze performance trade-offs and empirical comparisons",
                    "intent": "COMPARATIVE",
                    "priority": 1,
                    "dependencies": ["sq_001"],
                },
            ]
            content = json.dumps({"subqueries": mock_subs}, indent=2)
        else:
            # Dynamically extract evidence IDs from prompt to construct grounded claims
            evidence_ids = re.findall(r"\[EVIDENCE_ID:\s*(evidence_\d+)\]", prompt)

            if not evidence_ids:
                # Check for insufficient evidence
                mock_json = {
                    "summary": "Insufficient evidence to answer this reliably from the retrieved sources.",
                    "insufficient_evidence": True,
                    "sections": [],
                    "claims": [],
                    "limitations": ["No relevant evidence passages were found in the current index."],
                }
            else:
                claims = []
                for idx, ev_id in enumerate(evidence_ids[:4], start=1):
                    claims.append({
                        "claim_id": f"claim_{idx:03d}",
                        "text": f"Grounded research finding #{idx} supported directly by retrieved evidence {ev_id}.",
                        "evidence_ids": [ev_id],
                    })

                mock_json = {
                    "summary": "Synthesized grounded research findings based on retrieved literature.",
                    "insufficient_evidence": False,
                    "sections": [
                        {
                            "title": "Primary Research Findings",
                            "claims": claims,
                        }
                    ],
                    "claims": claims,
                    "limitations": [
                        "Synthesized strictly within the context of retrieved scientific documents."
                    ],
                }

            content = json.dumps(mock_json, indent=2)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        usage = LLMUsage(
            input_tokens=len(prompt) // 4,
            output_tokens=len(content) // 4,
            total_tokens=(len(prompt) + len(content)) // 4,
        )

        return LLMResponse(
            content=content,
            provider="mock",
            model=self._model_name,
            latency_ms=latency_ms,
            usage=usage,
            finish_reason="stop",
        )

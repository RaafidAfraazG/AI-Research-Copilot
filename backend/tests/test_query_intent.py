"""
Unit tests for layered Query Intent Classifier.
"""
import pytest

from llm.mock_provider import MockLLMProvider
from query.intent import QueryIntentClassifier
from query.models import QueryIntent


class TestQueryIntentClassifier:
    """Test suite verifying Level 1 deterministic rules and Level 2 LLM classification."""

    @pytest.mark.parametrize(
        "query,expected_intent",
        [
            ("What is BM25?", QueryIntent.FACTUAL),
            ("Define dense retrieval in neural search", QueryIntent.FACTUAL),
            ("Compare dense retrieval and BM25 for RAG systems", QueryIntent.COMPARATIVE),
            ("What are the trade-offs between cross-encoders and bi-encoders?", QueryIntent.COMPARATIVE),
            ("Why does dense retrieval struggle with exact keyword matching?", QueryIntent.CAUSAL),
            ("Why do large language models hallucinate?", QueryIntent.CAUSAL),
            ("How does reciprocal rank fusion work in hybrid search systems?", QueryIntent.EXPLANATORY),
            ("Explain the cross-encoder reranking mechanism", QueryIntent.EXPLANATORY),
            ("How to fine-tune an embedding model with hard negative mining?", QueryIntent.PROCEDURAL),
            ("How to design an enterprise hybrid search pipeline", QueryIntent.PROCEDURAL),
            ("Comprehensive survey of retrieval augmented generation architectures", QueryIntent.SURVEY),
            ("What are recent trends and state of the art advancements in 2024?", QueryIntent.TREND),
            ("Which retrieval benchmarks were used in DPR and how did ColBERT evaluate on them?", QueryIntent.MULTI_HOP),
        ],
    )
    def test_deterministic_intent_classification(
        self,
        query: str,
        expected_intent: QueryIntent,
    ) -> None:
        intent, conf = QueryIntentClassifier.classify_deterministic(query)
        assert intent == expected_intent
        assert conf >= 0.70

    def test_empty_query_returns_unknown(self) -> None:
        intent, conf = QueryIntentClassifier.classify_deterministic("")
        assert intent == QueryIntent.UNKNOWN
        assert conf == 0.0

    @pytest.mark.asyncio
    async def test_llm_classification_fallback(self) -> None:
        mock_llm = MockLLMProvider()
        query = "Evaluate trade-offs between dense and sparse representations"
        intent, conf = await QueryIntentClassifier.classify(
            query=query,
            llm_provider=mock_llm,
            use_llm=True,
        )
        assert intent == QueryIntent.COMPARATIVE
        assert conf > 0.80

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_deterministic(self) -> None:
        # Provider returning invalid json
        broken_llm = MockLLMProvider(custom_response="INVALID NON JSON OUTPUT")
        query = "Why does dense retrieval improve recall?"
        intent, conf = await QueryIntentClassifier.classify(
            query=query,
            llm_provider=broken_llm,
            use_llm=True,
        )
        assert intent == QueryIntent.CAUSAL

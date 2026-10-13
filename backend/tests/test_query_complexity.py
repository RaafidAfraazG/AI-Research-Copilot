"""
Unit tests for deterministic Query Complexity Analyzer.
"""
import pytest

from query.complexity import QueryComplexityAnalyzer
from query.models import QueryComplexity


class TestQueryComplexityAnalyzer:
    """Test suite verifying deterministic query complexity detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "What is RAG?",
            "What is BM25?",
            "Define dense retrieval",
            "What is Qdrant?",
            "Explain TF-IDF",
            "Define cross-encoder",
        ],
    )
    def test_simple_queries_classified_as_simple(self, query: str) -> None:
        complexity, req_decomp = QueryComplexityAnalyzer.analyze(query)
        assert complexity == QueryComplexity.SIMPLE
        assert req_decomp is False

    @pytest.mark.parametrize(
        "query",
        [
            "How does reciprocal rank fusion work in hybrid search systems?",
            "Why does dense retrieval struggle with exact keyword matching and out-of-vocabulary terms?",
            "How does RAG improve question answering and what are its primary limitations?",
            "How to fine-tune an embedding model with hard negative mining for domain retrieval?",
            "What are recent trends and state of the art advancements in neural reranking models in 2024?",
            "Comprehensive survey of retrieval augmented generation architectures and modular RAG patterns",
        ],
    )
    def test_moderate_queries_require_decomposition(self, query: str) -> None:
        complexity, req_decomp = QueryComplexityAnalyzer.analyze(query)
        assert complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX)
        assert req_decomp is True

    @pytest.mark.parametrize(
        "query",
        [
            "Compare dense retrieval and BM25 for RAG systems, explain their strengths and weaknesses, and identify when each performs better.",
            "Compare SPLADE sparse representations and ColBERT late interaction for search latency and memory overhead.",
            "Which retrieval benchmarks were used in the original Dense Passage Retrieval paper and how did ColBERT evaluate on them?",
            "How to design an enterprise hybrid search pipeline combining BM25, Qdrant dense vectors, and cross-encoder reranking, and what are the latency bottlenecks?",
        ],
    )
    def test_complex_queries_classified_as_complex(self, query: str) -> None:
        complexity, req_decomp = QueryComplexityAnalyzer.analyze(query)
        assert complexity == QueryComplexity.COMPLEX
        assert req_decomp is True

    def test_signal_extraction(self) -> None:
        query = "Compare dense retrieval and BM25 for RAG systems, and summarize recent trends in 2024?"
        signals = QueryComplexityAnalyzer.extract_signals(query)

        assert signals.token_count > 10
        assert signals.has_comparative is True
        assert signals.has_temporal is True
        assert signals.conjunction_count >= 1
        assert signals.entity_count >= 2
        assert signals.score >= 5.0

    def test_empty_query_returns_simple(self) -> None:
        complexity, req_decomp = QueryComplexityAnalyzer.analyze("   ")
        assert complexity == QueryComplexity.SIMPLE
        assert req_decomp is False

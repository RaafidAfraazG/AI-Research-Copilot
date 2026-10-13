"""
Manually curated research retrieval evaluation dataset for development benchmarking.
Contains 10 domain queries targeting the indexed scientific corpus.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationQuery(BaseModel):
    """Evaluation query paired with ground truth relevant source IDs."""
    query_id: str
    query_text: str
    relevant_source_ids: list[str] = Field(..., description="Target source IDs considered relevant")
    description: str = ""


RESEARCH_EVALUATION_DATASET: list[EvaluationQuery] = [
    EvaluationQuery(
        query_id="eval-01",
        query_text="What is Retrieval-Augmented Generation for knowledge intensive tasks?",
        relevant_source_ids=["2005.11401", "arxiv:2005.11401"],
        description="Core RAG architecture by Lewis et al.",
    ),
    EvaluationQuery(
        query_id="eval-02",
        query_text="How does Dense Passage Retrieval DPR work for open-domain question answering?",
        relevant_source_ids=["2004.04906", "arxiv:2004.04906"],
        description="DPR dual-encoder dense retrieval by Karpukhin et al.",
    ),
    EvaluationQuery(
        query_id="eval-03",
        query_text="How does Sentence-BERT use Siamese networks for cosine similarity embeddings?",
        relevant_source_ids=["1908.10084", "arxiv:1908.10084"],
        description="Sentence-BERT paper by Reimers & Gurevych.",
    ),
    EvaluationQuery(
        query_id="eval-04",
        query_text="What is the multi-head self-attention mechanism in the Transformer architecture?",
        relevant_source_ids=["1706.03762", "arxiv:1706.03762"],
        description="Original Transformer paper by Vaswani et al.",
    ),
    EvaluationQuery(
        query_id="eval-05",
        query_text="How does Reciprocal Rank Fusion RRF merge ranked lists without score calibration?",
        relevant_source_ids=["0905.0001", "arxiv:0905.0001"],
        description="RRF algorithm by Cormack et al.",
    ),
    EvaluationQuery(
        query_id="eval-06",
        query_text="How does contextualized late interaction scoring work in ColBERT?",
        relevant_source_ids=["2004.12832", "arxiv:2004.12832"],
        description="ColBERT late-interaction token retrieval by Khattab & Zaharia.",
    ),
    EvaluationQuery(
        query_id="eval-07",
        query_text="Listwise cross-encoder fine-tuning for LLM rerankers in medical procedure reranking",
        relevant_source_ids=["2608.09650", "arxiv:2608.09650"],
        description="Medical procedure cross-encoder reranker paper.",
    ),
    EvaluationQuery(
        query_id="eval-08",
        query_text="Fine-tuning BERT with triplet loss for semantic similarity search",
        relevant_source_ids=["1908.10084", "arxiv:1908.10084"],
        description="Siamese and triplet objective functions in SBERT.",
    ),
    EvaluationQuery(
        query_id="eval-09",
        query_text="Combining BM25 lexical keyword matching with dense passage retrieval",
        relevant_source_ids=["0905.0001", "arxiv:0905.0001", "2004.04906", "arxiv:2004.04906"],
        description="Hybrid retrieval fusion concepts.",
    ),
    EvaluationQuery(
        query_id="eval-10",
        query_text="Autoregressive retrieval augmentation for context-aware generation",
        relevant_source_ids=["2506.06962", "arxiv:2506.06962", "2005.11401", "arxiv:2005.11401"],
        description="Autoregressive RAG and retrieval augmentation.",
    ),
]

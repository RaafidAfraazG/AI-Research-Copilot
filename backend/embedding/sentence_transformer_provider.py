"""
Local embedding provider using sentence-transformers (e.g. all-MiniLM-L6-v2).

Loading is lazy but FAIL-CLOSED: if the model cannot be loaded an
EmbeddingModelLoadError is raised. No mock vectors are ever substituted.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

from embedding.errors import EmbeddingModelLoadError

logger = structlog.get_logger(__name__)


class SentenceTransformerProvider:
    """
    Local embedding provider backed by huggingface/sentence-transformers.

    Features:
    - Lazy model loading on first use (``dimension``/``revision`` also trigger it,
      so identity always reflects the actually loaded model)
    - L2-normalized embeddings (configurable) for cosine similarity via dot product
    - Async execution via asyncio.to_thread to keep the FastAPI event loop unblocked
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
        dimension: int = 384,
        normalize_embeddings: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._dimension = dimension  # provisional until the model is loaded
        self.normalize_embeddings = normalize_embeddings
        self._model: Any = None
        self._revision: str | None = None

    @property
    def dimension(self) -> int:
        self._get_or_load_model()
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str | None:
        self._get_or_load_model()
        return self._revision

    def _get_or_load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingModelLoadError(
                "sentence-transformers is not installed; cannot load the configured "
                "embedding model. Install it, or explicitly set EMBEDDING_PROVIDER=mock "
                "for tests/experiments."
            ) from e
        try:
            logger.info("loading_sentence_transformer_model", model_name=self._model_name)
            model = SentenceTransformer(self._model_name, device=self._device)
        except Exception as e:
            # Safe diagnostics only: exception class, never paths or secrets.
            logger.error(
                "sentence_transformer_load_failed",
                model_name=self._model_name,
                error_type=type(e).__name__,
            )
            raise EmbeddingModelLoadError(
                f"Failed to load embedding model '{self._model_name}' ({type(e).__name__}). "
                "Dense retrieval and indexing are unavailable; mock embeddings are NOT "
                "substituted automatically."
            ) from e

        dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        dim = dim_fn()
        if dim:
            self._dimension = int(dim)
        self._revision = self._discover_revision(model)
        self._model = model
        return model

    @staticmethod
    def _discover_revision(model: Any) -> str | None:
        """Best-effort model commit hash (config first, then the HF cache's refs/main)."""
        try:
            config = model[0].auto_model.config
            rev = getattr(config, "_commit_hash", None)
            if rev:
                return str(rev)
            repo = str(getattr(config, "_name_or_path", "") or "")
            if repo and "/" in repo and not repo.startswith((".", "/")):
                from pathlib import Path

                from huggingface_hub import constants

                ref = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "refs" / "main"
                if ref.exists():
                    return ref.read_text(encoding="utf-8").strip() or None
        except Exception:
            return None
        return None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_or_load_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        results = self.embed_texts([text])
        return results[0] if results else [0.0] * self.dimension

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_texts, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

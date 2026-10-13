"""
Application settings loaded from environment variables and the project-root .env file.

Design notes:
- All configuration comes from environment variables or a .env file.
- The .env file is located at the monorepo root (one level above backend/).
- This means `cp .env.example .env` at the project root is the only setup step.
- Credentials are NEVER hard-coded. Defaults are safe for local dev only.
- The Settings object is cached via @lru_cache. Call get_settings.cache_clear()
  in tests to avoid stale state.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root from this file's location:
# backend/core/config.py -> backend/core/ -> backend/ -> project-root/
_ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
_ENV_FILE: str = str(_ROOT_DIR / ".env")


class Settings(BaseSettings):
    """
    Central settings model for the AI Research Copilot backend.
    All fields correspond to environment variables documented in .env.example.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",          # Ignore Docker-specific vars not used by the app
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = True
    app_log_level: str = "INFO"
    backend_port: int = 8000
    secret_key: str = "change_me_to_a_random_secret_in_production"

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: list[str] | str = ["http://localhost:5173", "http://localhost:3000"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: object) -> list[str]:
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.startswith("[") and v_str.endswith("]"):
                import json
                try:
                    return json.loads(v_str)
                except Exception:
                    pass
            return [i.strip() for i in v_str.split(",") if i.strip()]
        elif isinstance(v, (list, tuple)):
            return [str(i).strip() for i in v]
        return v  # type: ignore[return-value]

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "copilot_user"
    postgres_password: str = "change_me_in_production"
    postgres_db: str = "ai_research_copilot"
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20
    postgres_pool_timeout: int = 30

    # ── OpenSearch ───────────────────────────────────────────────────────────
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_user: str = "admin"
    opensearch_password: str = "change_me_in_production"
    opensearch_use_ssl: bool = False
    opensearch_index_name: str = "documents"
    opensearch_timeout: int = 10
    # Probe the (unused in local setups) OpenSearch / Qdrant server / Redis in /ready.
    # Informational only: never causes a 503.
    readiness_probe_optional: bool = False

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "document_chunks"
    qdrant_timeout: int = 10

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None
    redis_db: int = 0
    cache_ttl_seconds: int = 3600

    # ── LLM Provider ─────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic", "ollama", "gemini", "mock"] = "gemini"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-haiku-20240307"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # ── Gemini (primary free-tier provider) ───────────────────────────────────
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"

    # ── Paid-provider safety guard ────────────────────────────────────────────
    # Set to false (default) to block providers that require paid API keys.
    # Gemini free tier and MockLLM are always allowed regardless of this flag.
    llm_allow_paid_provider: bool = False

    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # ── Embeddings ───────────────────────────────────────────────────────────
    # "mock" is an explicit opt-in for tests/experiments. It is never selected
    # automatically when the real provider fails to load.
    embedding_provider: Literal["local", "openai", "mock"] = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_normalize: bool = True

    # ── Data root ────────────────────────────────────────────────────────────
    # Canonical data directory (indexes, corpus, backups). Relative values are
    # resolved against the repository root, never the current working directory.
    # Default: <repo>/data
    data_dir: str | None = None

    # ── Retrieval Pipeline ───────────────────────────────────────────────────
    retrieval_bm25_top_n: int = 30
    retrieval_dense_top_n: int = 30
    retrieval_top_k: int = 10
    rrf_k: int = 60
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # ── Reranking & Evidence Selection ───────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"
    reranker_max_length: int = 512
    reranker_batch_size: int = 16
    reranker_candidate_k: int = 25
    reranker_candidates: int = 25  # alias for backward compatibility
    reranker_fallback_mode: Literal["hybrid_score", "error"] = "hybrid_score"
    evidence_top_k: int = 6
    evidence_min_score: float | None = None
    evidence_max_per_document: int = 2
    evidence_similarity_threshold: float = 0.85

    # ── Query Intelligence & Decomposition ───────────────────────────────────
    enable_decomposition: bool = True
    decomposition_mode: Literal["auto", "llm", "deterministic"] = "auto"
    max_subqueries: int = 5
    max_total_evidence: int = 12
    max_evidence_per_subquery: int = 4



    # ── Source Connectors ────────────────────────────────────────────────────
    arxiv_max_results: int = 10
    arxiv_enabled: bool = True
    semantic_scholar_api_key: str | None = None
    semantic_scholar_max_results: int = 10
    semantic_scholar_enabled: bool = True
    web_fetcher_enabled: bool = True
    web_fetcher_timeout_seconds: int = 10
    github_token: str | None = None
    github_enabled: bool = False

    # ── Derived properties ───────────────────────────────────────────────────

    @property
    def database_url(self) -> str:
        """Async PostgreSQL connection URL for SQLAlchemy + asyncpg."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Synchronous PostgreSQL URL — used by Alembic migrations."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("app_log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"app_log_level must be one of {valid}, got '{v}'")
        return upper

    @model_validator(mode="after")
    def validate_production_secrets(self) -> Settings:
        """Fail fast if production secrets are still set to defaults."""
        if self.app_env == "production":
            if self.secret_key == "change_me_to_a_random_secret_in_production":
                raise ValueError(
                    "SECRET_KEY must be changed for production deployments. "
                    "Set a strong random value in your environment."
                )
        return self


@lru_cache
def get_settings() -> Settings:
    """
    Return cached application settings.

    The result is cached after first call for performance.
    In tests, call `get_settings.cache_clear()` before each test that
    modifies environment variables to prevent stale settings.
    """
    return Settings()

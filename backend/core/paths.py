"""
Canonical filesystem layout, independent of the process working directory.

    <data_root>/                 default: <repo>/data  (override with DATA_DIR)
      corpus/corpus.json         authoritative source documents for index builds
      evaluation/                benchmark datasets (never modified by indexing)
      indexes/
        ACTIVE                   pointer to the active index generation
        <generation_id>/
          manifest.json          embedding/chunking/count manifest
          bm25_index.json        local BM25 store
          qdrant/                embedded Qdrant storage
      backups/<timestamp>/       snapshots taken before migrations/rebuilds
"""
from __future__ import annotations

from pathlib import Path

from core.config import Settings, get_settings

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def resolve_data_root(settings: Settings | None = None) -> Path:
    """Return the absolute canonical data root (never cwd-relative)."""
    st = settings or get_settings()
    raw = (st.data_dir or "").strip()
    if not raw:
        return REPO_ROOT / "data"
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def corpus_file(settings: Settings | None = None) -> Path:
    return resolve_data_root(settings) / "corpus" / "corpus.json"


def indexes_root(settings: Settings | None = None) -> Path:
    return resolve_data_root(settings) / "indexes"


def backups_root(settings: Settings | None = None) -> Path:
    return resolve_data_root(settings) / "backups"

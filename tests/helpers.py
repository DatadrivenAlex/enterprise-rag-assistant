"""Shared builders for offline tests."""

from __future__ import annotations

import uuid
from pathlib import Path

from enterprise_rag.config import Settings

REPO_DATA = Path(__file__).resolve().parents[1] / "data"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": tmp_path,
        "vector_persist": False,
        "vector_backend": "faiss",
        "auto_ingest_dir": None,
        "embedding_provider": "hash",
        "chat_provider": "extractive",
        "collection_name": "col" + uuid.uuid4().hex[:12],
        "chroma_persist_dir": tmp_path / "chroma",
        "faiss_index_path": tmp_path / "faiss",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]

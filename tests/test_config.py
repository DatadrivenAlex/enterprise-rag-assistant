from pathlib import Path

import pytest
from pydantic import ValidationError

from enterprise_rag.config import Settings


def test_rag_prefix_and_blank_secret(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "faiss")
    settings = Settings(
        data_dir=tmp_path,
        auto_ingest_dir=None,
        openai_api_key="   ",
        _env_file=None,
    )
    assert settings.top_k == 3
    assert settings.vector_backend == "faiss"
    assert settings.openai_api_key is None


def test_chunk_overlap_must_fit(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, auto_ingest_dir=None, chunk_size=100, chunk_overlap=100)


def test_openai_chat_requires_a_key(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, auto_ingest_dir=None, chat_provider="openai")

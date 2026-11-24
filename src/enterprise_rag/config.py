"""Runtime configuration loaded from the environment and an optional .env file."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ChatProvider = Literal["extractive", "openai"]
EmbeddingProvider = Literal["hash", "openai"]
VectorBackend = Literal["chroma", "faiss"]

_CANARY_RE = re.compile(r"[A-Za-z0-9_\-]{8,64}")
_COLLECTION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]")


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


class Settings(BaseSettings):
    """Application settings.

    Environment variables use the ``RAG_`` prefix, for example ``RAG_TOP_K``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Enterprise RAG Assistant"
    environment: str = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    cors_origins: str = "*"

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    chat_provider: ChatProvider = "extractive"
    embedding_provider: EmbeddingProvider = "hash"
    embedding_dimension: int = Field(default=384, ge=32, le=3072)

    vector_backend: VectorBackend = "chroma"
    retrieval_mode: Literal["hybrid", "vector"] = "hybrid"
    vector_persist: bool = True
    data_dir: Path = Field(default_factory=default_data_dir)
    chroma_persist_dir: Path | None = None
    faiss_index_path: Path | None = None
    collection_name: str = "enterprise_docs"

    chunk_size: int = Field(default=800, ge=40, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    top_k: int = Field(default=4, ge=1, le=20)
    max_query_chars: int = Field(default=4000, ge=100, le=20000)
    max_upload_bytes: int = Field(default=1_000_000, ge=1_000, le=20_000_000)
    auto_ingest_dir: str | None = "sample_docs"
    canary_token: str = "RAG_CANARY_7f3a9c"

    @field_validator("api_key", "openai_api_key", "auto_ingest_dir", mode="before")
    @classmethod
    def blank_string_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("canary_token")
    @classmethod
    def canary_is_safe_token(cls, value: str) -> str:
        if not _CANARY_RE.fullmatch(value):
            raise ValueError("canary_token must be 8-64 characters from [A-Za-z0-9_-]")
        return value

    @field_validator("collection_name")
    @classmethod
    def collection_name_is_portable(cls, value: str) -> str:
        if not _COLLECTION_RE.fullmatch(value):
            raise ValueError(
                "collection_name must be 3-512 characters from [A-Za-z0-9._-] "
                "and start and end with an alphanumeric character"
            )
        return value

    @field_validator("log_level")
    @classmethod
    def log_level_upper(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def chunk_overlap_fits(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.chat_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "RAG_OPENAI_API_KEY is required when RAG_CHAT_PROVIDER=openai"
            )
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "RAG_OPENAI_API_KEY is required when RAG_EMBEDDING_PROVIDER=openai"
            )
        return self

    @property
    def resolved_chroma_dir(self) -> Path:
        if self.chroma_persist_dir is not None:
            return self.chroma_persist_dir
        return self.data_dir / "chroma"

    @property
    def resolved_faiss_dir(self) -> Path:
        if self.faiss_index_path is not None:
            return self.faiss_index_path
        return self.data_dir / "faiss"

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

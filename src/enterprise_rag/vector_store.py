"""Vector store adapters for ChromaDB and FAISS, plus offline hash embeddings.

Hash embeddings are deterministic and need no network. They are the default so
the service and the test suite run without an API key. Set
``RAG_EMBEDDING_PROVIDER=openai`` for production semantic embeddings.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from enterprise_rag.config import Settings
from enterprise_rag.document_loader import TextChunk

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

Metadata = dict[str, str | int | float | bool]


@dataclass
class ScoredChunk:
    id: str
    text: str
    metadata: Metadata
    score: float


class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def upsert(self, chunks: list[TextChunk]) -> int: ...

    def delete_source(self, source: str) -> None: ...

    def similarity_search(self, query: str, k: int) -> list[ScoredChunk]: ...

    def count(self) -> int: ...

    def list_sources(self) -> list[str]: ...


def hash_embed(text: str, dimension: int) -> list[float]:
    """Bag-of-hashed-tokens embedding, L2-normalized for cosine comparison."""
    vector = [0.0] * dimension
    tokens = re.findall(r"[a-z0-9]+", text.lower()) or ["empty"]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class HashEmbedder:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [hash_embed(text, self.dimension) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return hash_embed(text, self.dimension)


class OpenAIEmbedder:
    """Embeddings via the OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        dimension: int,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import httpx

        vectors: list[list[float]] = []
        try:
            with httpx.Client(timeout=self.timeout) as client:
                for start in range(0, len(texts), 64):
                    batch = texts[start : start + 64]
                    response = client.post(
                        f"{self.base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"model": self.model, "input": batch, "dimensions": self.dimension},
                    )
                    response.raise_for_status()
                    rows = response.json()["data"]
                    rows.sort(key=lambda row: row["index"])
                    vectors.extend(row["embedding"] for row in rows)
        except httpx.HTTPError as exc:
            raise RuntimeError("Embedding request failed") from exc
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashEmbedder(dimension=settings.embedding_dimension)
    if not settings.openai_api_key:
        raise ValueError("RAG_OPENAI_API_KEY is required when RAG_EMBEDDING_PROVIDER=openai")
    return OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        base_url=settings.openai_base_url,
        dimension=settings.embedding_dimension,
    )


def _chroma_embedding_function(embedder: Embedder):
    import chromadb.utils.embedding_functions as embedding_functions
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    @embedding_functions.register_embedding_function
    class EnterpriseEmbeddingFunction(EmbeddingFunction[Documents]):
        def __init__(self, dimension: int = embedder.dimension) -> None:
            self.dimension = dimension
            self._embedder = embedder

        def __call__(self, input: Documents) -> Embeddings:
            return self._embedder.embed_documents(list(input))

        @staticmethod
        def name() -> str:
            return "enterprise-hash-or-configured"

        def get_config(self) -> dict[str, int]:
            return {"dimension": self.dimension}

        @staticmethod
        def build_from_config(config: dict[str, int]) -> EnterpriseEmbeddingFunction:
            return EnterpriseEmbeddingFunction(dimension=int(config.get("dimension", embedder.dimension)))

    return EnterpriseEmbeddingFunction(dimension=embedder.dimension)


class ChromaVectorStore:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._embedder = embedder
        chroma_settings = ChromaSettings(anonymized_telemetry=False)
        if settings.vector_persist:
            settings.resolved_chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(settings.resolved_chroma_dir),
                settings=chroma_settings,
            )
        else:
            self._client = chromadb.EphemeralClient(settings=chroma_settings)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=_chroma_embedding_function(embedder),
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[_chroma_metadata(chunk.metadata) for chunk in chunks],
        )
        return len(chunks)

    def delete_source(self, source: str) -> None:
        existing = self._collection.get(where={"source": source}, include=["metadatas"])
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)

    def similarity_search(self, query: str, k: int) -> list[ScoredChunk]:
        total = self.count()
        if total == 0 or k <= 0:
            return []
        result = self._collection.query(
            query_texts=[query],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[ScoredChunk] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 - float(distance)
            hits.append(
                ScoredChunk(
                    id=str(chunk_id),
                    text=document or "",
                    metadata=dict(metadata or {}),
                    score=max(0.0, min(1.0, score)),
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def list_sources(self) -> list[str]:
        if self.count() == 0:
            return []
        payload = self._collection.get(include=["metadatas"])
        sources = {
            str(metadata["source"])
            for metadata in payload.get("metadatas") or []
            if metadata and metadata.get("source")
        }
        return sorted(sources)


class FaissVectorStore:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        import faiss

        self._faiss = faiss
        self._embedder = embedder
        self._dimension = embedder.dimension
        self._persist = settings.vector_persist
        self._directory = settings.resolved_faiss_dir
        self._index_file = self._directory / "index.faiss"
        self._meta_file = self._directory / "meta.json"
        self._items: dict[int, dict[str, object]] = {}
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))
        if self._persist:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._load()

    def upsert(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0
        vectors = np.asarray(self._embedder.embed_documents([chunk.text for chunk in chunks]), dtype=np.float32)
        ids = np.asarray([_faiss_id(chunk.id) for chunk in chunks], dtype=np.int64)
        existing = np.asarray([item_id for item_id in ids if item_id in self._items], dtype=np.int64)
        if len(existing):
            self._index.remove_ids(existing)
        self._index.add_with_ids(np.ascontiguousarray(vectors), ids)
        for chunk, item_id in zip(chunks, ids):
            self._items[int(item_id)] = {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": dict(chunk.metadata),
            }
        self._save()
        return len(chunks)

    def delete_source(self, source: str) -> None:
        doomed = [
            item_id
            for item_id, item in self._items.items()
            if dict(item["metadata"]).get("source") == source
        ]
        if not doomed:
            return
        self._index.remove_ids(np.asarray(doomed, dtype=np.int64))
        for item_id in doomed:
            self._items.pop(item_id, None)
        self._save()

    def similarity_search(self, query: str, k: int) -> list[ScoredChunk]:
        if not self._items or k <= 0:
            return []
        vector = np.asarray([self._embedder.embed_query(query)], dtype=np.float32)
        scores, ids = self._index.search(np.ascontiguousarray(vector), min(k, len(self._items)))
        hits: list[ScoredChunk] = []
        for score, item_id in zip(scores[0], ids[0]):
            if int(item_id) < 0:
                continue
            item = self._items.get(int(item_id))
            if item is None:
                continue
            hits.append(
                ScoredChunk(
                    id=str(item["id"]),
                    text=str(item["text"]),
                    metadata=dict(item["metadata"]),  # type: ignore[arg-type]
                    score=float(score),
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits

    def count(self) -> int:
        return len(self._items)

    def list_sources(self) -> list[str]:
        sources = {
            str(dict(item["metadata"]).get("source"))
            for item in self._items.values()
            if dict(item["metadata"]).get("source")
        }
        return sorted(sources)

    def _save(self) -> None:
        if not self._persist:
            return
        self._faiss.write_index(self._index, str(self._index_file))
        payload = {
            "dimension": self._dimension,
            "items": {str(item_id): item for item_id, item in self._items.items()},
        }
        self._meta_file.write_text(json.dumps(payload), encoding="utf-8")

    def _load(self) -> None:
        if not self._meta_file.exists() or not self._index_file.exists():
            return
        payload = json.loads(self._meta_file.read_text(encoding="utf-8"))
        if int(payload.get("dimension", 0)) != self._dimension:
            raise ValueError("FAISS index dimension does not match RAG_EMBEDDING_DIMENSION")
        self._index = self._faiss.read_index(str(self._index_file))
        self._items = {int(item_id): item for item_id, item in payload.get("items", {}).items()}


def build_vector_store(settings: Settings, embedder: Embedder | None = None) -> VectorStore:
    selected = embedder or build_embedder(settings)
    if settings.vector_backend == "faiss":
        return FaissVectorStore(settings, selected)
    return ChromaVectorStore(settings, selected)


def _chroma_metadata(metadata: Metadata) -> Metadata:
    clean: Metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _faiss_id(chunk_id: str) -> int:
    digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF

"""Deterministic BM25 retrieval fused with vector ranks via reciprocal rank fusion."""

from __future__ import annotations
import heapq
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace
from threading import RLock
from enterprise_rag.document_loader import TextChunk
from enterprise_rag.vector_store import ScoredChunk, VectorStore

_TOKEN = re.compile(r"[^\W_]+(?:[-_.][^\W_]+)*", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


class BM25:
    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self.chunks = {c.id: c for c in chunks}
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        lengths: dict[str, int] = {}
        for c in self.chunks.values():
            tokens = _tokens(c.text)
            lengths[c.id] = len(tokens)
            for token, count in Counter(tokens).items():
                self.postings[token][c.id] = count
        avg = sum(lengths.values()) / max(1, len(lengths)) or 1
        self.norms = {cid: 1.2 * (0.25 + 0.75 * n / avg) for cid, n in lengths.items()}

    def search(self, query: str, limit: int) -> list[ScoredChunk]:
        scores: dict[str, float] = defaultdict(float)
        for token in dict.fromkeys(_tokens(query)):
            posting = self.postings.get(token, {})
            if not posting:
                continue
            idf = math.log1p(
                (len(self.chunks) - len(posting) + 0.5) / (len(posting) + 0.5)
            )
            for cid, tf in posting.items():
                scores[cid] += idf * (tf * 2.2) / (tf + self.norms[cid])
        return [
            self.chunks[cid]
            for cid in heapq.nsmallest(limit, scores, key=lambda x: (-scores[x], x))
        ]


def fuse(rankings: list[list[ScoredChunk]], limit: int) -> list[ScoredChunk]:
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, 1):
            chunks[chunk.id] = chunk
            scores[chunk.id] += 1 / (60 + rank)
    return [
        replace(chunks[cid], score=scores[cid])
        for cid in heapq.nsmallest(limit, scores, key=lambda x: (-scores[x], x))
    ]


class HybridStore:
    """Lazy lexical cache; ingestion invalidates it atomically within one service process."""

    def __init__(self, vector: VectorStore) -> None:
        self._vector = vector
        self._lexical: BM25 | None = None
        self._lock = RLock()

    def upsert(self, chunks: list[TextChunk]) -> int:
        with self._lock:
            self._lexical = None
            return self._vector.upsert(chunks)

    def delete_source(self, source: str) -> None:
        with self._lock:
            self._lexical = None
            self._vector.delete_source(source)

    def replace_sources(self, sources: list[str], chunks: list[TextChunk]) -> None:
        with self._lock:
            for source in sources:
                self._vector.delete_source(source)
            self._lexical = None
            self._vector.upsert(chunks)

    def similarity_search(self, query: str, k: int) -> list[ScoredChunk]:
        if k <= 0:
            return []
        depth = max(20, 3 * k)
        with self._lock:
            if self._lexical is None:
                self._lexical = BM25(self._vector.all_chunks())
            lexical = self._lexical.search(query, depth)
            dense = self._vector.similarity_search(query, depth)
        return fuse([lexical, [h for h in dense if h.score > 0]], k)

    def count(self) -> int:
        return self._vector.count()

    def list_sources(self) -> list[str]:
        return self._vector.list_sources()

    def all_chunks(self) -> list[ScoredChunk]:
        return self._vector.all_chunks()

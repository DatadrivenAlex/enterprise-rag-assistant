from pathlib import Path

import pytest

from enterprise_rag.document_loader import TextChunk
from enterprise_rag.hybrid import BM25, HybridStore, fuse
from enterprise_rag.vector_store import ScoredChunk, build_vector_store
from tests.helpers import make_settings

HANDBOOK = "Employees receive 28 paid vacation days per year. Submit requests in the HR portal."
SECURITY = (
    "Report security incidents to security@helios.example within 1 hour of discovery."
)


def _chunks() -> list[TextChunk]:
    return [
        TextChunk(
            id="handbook:0",
            text=HANDBOOK,
            metadata={
                "source": "employee_handbook.md",
                "title": "Employee Handbook",
                "chunk_index": 0,
            },
        ),
        TextChunk(
            id="security:0",
            text=SECURITY,
            metadata={
                "source": "information_security_policy.md",
                "title": "Security",
                "chunk_index": 0,
            },
        ),
    ]


@pytest.mark.parametrize("backend", ["chroma", "faiss"])
def test_similarity_search_ranks_the_matching_document(
    tmp_path: Path, backend: str
) -> None:
    settings = make_settings(tmp_path, vector_backend=backend)
    store = build_vector_store(settings)
    assert store.upsert(_chunks()) == 2
    hits = store.similarity_search(
        "How many paid vacation days do employees receive?", k=2
    )
    assert hits[0].metadata["source"] == "employee_handbook.md"
    assert hits[0].score >= hits[1].score
    assert "28" in hits[0].text
    assert store.list_sources() == [
        "employee_handbook.md",
        "information_security_policy.md",
    ]


@pytest.mark.parametrize("backend", ["chroma", "faiss"])
def test_delete_source_removes_only_that_source(tmp_path: Path, backend: str) -> None:
    settings = make_settings(tmp_path, vector_backend=backend)
    store = build_vector_store(settings)
    store.upsert(_chunks())
    store.delete_source("employee_handbook.md")
    assert store.count() == 1
    assert store.list_sources() == ["information_security_policy.md"]


@pytest.mark.parametrize("backend", ["chroma", "faiss"])
def test_persisted_index_reloads(tmp_path: Path, backend: str) -> None:
    settings = make_settings(tmp_path, vector_backend=backend, vector_persist=True)
    store = build_vector_store(settings)
    store.upsert(_chunks()[:1])
    del store
    reloaded = build_vector_store(settings)
    hits = reloaded.similarity_search("paid vacation days", k=1)
    assert reloaded.count() == 1
    assert hits and "28" in hits[0].text


def test_bm25_finds_exact_rare_terms_without_vector_semantics() -> None:
    docs = [
        ScoredChunk(
            "a", "The incident uses PostgreSQL pgvector and HNSW", {"source": "a.md"}, 0
        ),
        ScoredChunk("b", "Employees receive vacation days", {"source": "b.md"}, 0),
    ]
    hits = BM25(docs).search("PostgreSQL pgvector", 2)
    assert hits[0].id == "a"


def test_rrf_combines_ranks_not_incomparable_scores() -> None:
    a = ScoredChunk("a", "A", {"source": "a"}, 100.0)
    b = ScoredChunk("b", "B", {"source": "b"}, 0.01)
    result = fuse([[a, b], [b, a]], 2)
    assert {item.id for item in result} == {"a", "b"}
    assert result[0].score == result[1].score


class TinyVectorStore:
    def __init__(self):
        self.items = {}

    def upsert(self, chunks):
        for chunk in chunks:
            self.items[chunk.id] = chunk
        return len(chunks)

    def delete_source(self, source):
        self.items = {
            k: v for k, v in self.items.items() if v.metadata.get("source") != source
        }

    def similarity_search(self, query, k):
        return []

    def count(self):
        return len(self.items)

    def list_sources(self):
        return sorted({v.metadata["source"] for v in self.items.values()})

    def all_chunks(self):
        return [ScoredChunk(v.id, v.text, v.metadata, 0) for v in self.items.values()]


def test_hybrid_index_invalidates_after_source_replacement() -> None:
    vector = TinyVectorStore()
    store = HybridStore(vector)
    old = TextChunk("same", "stale keyword", {"source": "docs.md"})
    new = TextChunk("same", "fresh keyword", {"source": "docs.md"})
    store.upsert([old])
    assert store.similarity_search("stale", 1)
    store.replace_sources(["docs.md"], [new])
    assert store.similarity_search("stale", 1) == []
    assert store.similarity_search("fresh", 1)[0].text == "fresh keyword"

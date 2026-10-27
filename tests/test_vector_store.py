from pathlib import Path

import pytest

from enterprise_rag.document_loader import TextChunk
from enterprise_rag.vector_store import build_vector_store
from tests.helpers import make_settings

HANDBOOK = "Employees receive 28 paid vacation days per year. Submit requests in the HR portal."
SECURITY = "Report security incidents to security@helios.example within 1 hour of discovery."


def _chunks() -> list[TextChunk]:
    return [
        TextChunk(
            id="handbook:0",
            text=HANDBOOK,
            metadata={"source": "employee_handbook.md", "title": "Employee Handbook", "chunk_index": 0},
        ),
        TextChunk(
            id="security:0",
            text=SECURITY,
            metadata={"source": "information_security_policy.md", "title": "Security", "chunk_index": 0},
        ),
    ]


@pytest.mark.parametrize("backend", ["chroma", "faiss"])
def test_similarity_search_ranks_the_matching_document(tmp_path: Path, backend: str) -> None:
    settings = make_settings(tmp_path, vector_backend=backend)
    store = build_vector_store(settings)
    assert store.upsert(_chunks()) == 2
    hits = store.similarity_search("How many paid vacation days do employees receive?", k=2)
    assert hits[0].metadata["source"] == "employee_handbook.md"
    assert hits[0].score >= hits[1].score
    assert "28" in hits[0].text
    assert store.list_sources() == ["employee_handbook.md", "information_security_policy.md"]


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

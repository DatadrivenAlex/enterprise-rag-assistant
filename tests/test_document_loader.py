from pathlib import Path

import pytest

from enterprise_rag.document_loader import IngestPathError, load_path, resolve_ingest_path


def test_chunks_include_source_metadata(tmp_path: Path) -> None:
    document = tmp_path / "policy.md"
    document.write_text(
        "# Leave Policy\n\n" + ("Employees receive 28 paid vacation days per year. " * 30),
        encoding="utf-8",
    )
    loaded = load_path(document, chunk_size=180, chunk_overlap=20)
    assert loaded.files == 1
    assert len(loaded.chunks) > 1
    first = loaded.chunks[0]
    assert first.metadata["source"] == "policy.md"
    assert first.metadata["title"] == "Leave Policy"
    assert first.metadata["chunk_index"] == 0
    assert first.metadata["file_type"] == "md"
    assert isinstance(first.metadata["start_char"], int)
    assert first.id.endswith(":0")
    assert loaded.chunks[1].metadata["chunk_index"] == 1


def test_directory_load_skips_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("Support tickets are retained for 36 months after closure.", encoding="utf-8")
    (tmp_path / "picture.bin").write_bytes(b"\x00\x01not text")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "extra.md").write_text("# Extra\n\nRemote work is limited to 3 days per week.", encoding="utf-8")
    loaded = load_path(tmp_path, chunk_size=400, chunk_overlap=0)
    sources = {chunk.metadata["source"] for chunk in loaded.chunks}
    assert loaded.files == 2
    assert sources == {"notes.txt", "nested/extra.md"}


def test_ingest_path_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(IngestPathError):
        resolve_ingest_path(tmp_path, "../etc")
    with pytest.raises(IngestPathError):
        resolve_ingest_path(tmp_path, "/etc")


def test_empty_file_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")
    loaded = load_path(tmp_path, chunk_size=200, chunk_overlap=0)
    assert loaded.files == 0
    assert loaded.chunks == []

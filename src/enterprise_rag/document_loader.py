"""Load markdown and text files and split them into chunks with source metadata."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


class IngestPathError(ValueError):
    """Raised when an ingest path escapes the configured data directory."""


@dataclass
class TextChunk:
    id: str
    text: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass
class LoadedDocuments:
    files: int
    chunks: list[TextChunk]


def resolve_ingest_path(data_dir: Path, raw: str) -> Path:
    """Resolve ``raw`` under ``data_dir`` and reject path traversal."""
    root = data_dir.resolve()
    candidate = Path(raw)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise IngestPathError("Ingest path must stay inside the configured data directory")
    return resolved


def load_path(
    path: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_file_bytes: int = 1_000_000,
) -> LoadedDocuments:
    path = path.resolve()
    if path.is_file():
        chunk = _load_file(path, path.parent, chunk_size, chunk_overlap, max_file_bytes)
        return LoadedDocuments(files=1 if chunk else 0, chunks=chunk)
    if path.is_dir():
        return _load_directory(path, chunk_size, chunk_overlap, max_file_bytes)
    raise FileNotFoundError(path)


def _load_directory(
    root: Path,
    chunk_size: int,
    chunk_overlap: int,
    max_file_bytes: int,
) -> LoadedDocuments:
    files = sorted(
        item
        for item in root.rglob("*")
        if item.is_file()
        and item.suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part.startswith(".") for part in item.relative_to(root).parts)
    )
    chunks: list[TextChunk] = []
    used = 0
    for file_path in files:
        loaded = _load_file(file_path, root, chunk_size, chunk_overlap, max_file_bytes)
        if loaded:
            used += 1
            chunks.extend(loaded)
    return LoadedDocuments(files=used, chunks=chunks)


def _load_file(
    file_path: Path,
    root: Path,
    chunk_size: int,
    chunk_overlap: int,
    max_file_bytes: int,
) -> list[TextChunk]:
    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return []
    if file_path.stat().st_size > max_file_bytes:
        return []
    text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    source = file_path.relative_to(root).as_posix()
    title = _title(text, file_path)
    doc_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    documents = splitter.create_documents([text])
    chunks: list[TextChunk] = []
    for index, document in enumerate(documents):
        content = document.page_content.strip()
        if not content:
            continue
        start = int(document.metadata.get("start_index", 0))
        chunks.append(
            TextChunk(
                id=f"{doc_id}:{index}",
                text=content,
                metadata={
                    "source": source,
                    "title": title,
                    "doc_id": doc_id,
                    "chunk_index": index,
                    "start_char": start,
                    "end_char": start + len(content),
                    "file_type": file_path.suffix.lower().lstrip("."),
                },
            )
        )
    return chunks


def _title(text: str, file_path: Path) -> str:
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()[:200]
    return file_path.stem.replace("_", " ").replace("-", " ").title()

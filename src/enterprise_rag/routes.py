"""HTTP routes for ingest, query, evaluation, and source listing."""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from enterprise_rag.config import Settings
from enterprise_rag.document_loader import SUPPORTED_SUFFIXES, IngestPathError, resolve_ingest_path
from enterprise_rag.rag_pipeline import GenerationError, KnowledgeService
from enterprise_rag.schemas import (
    EvaluateRequest,
    EvaluationResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceListResponse,
)

router = APIRouter(prefix="/api/v1", tags=["rag"])


def get_service(request: Request) -> KnowledgeService:
    return request.app.state.service


def get_settings_from_app(request: Request) -> Settings:
    return request.app.state.settings


def require_api_key(
    request: Request,
    settings: Settings = Depends(get_settings_from_app),
) -> None:
    expected = settings.api_key
    if not expected:
        return
    provided = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


@router.get("/sources", response_model=SourceListResponse, dependencies=[Depends(require_api_key)])
def list_sources(service: KnowledgeService = Depends(get_service)) -> SourceListResponse:
    return SourceListResponse(sources=service.store.list_sources(), chunks=service.store.count())


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest(body: IngestRequest, service: KnowledgeService = Depends(get_service)) -> IngestResponse:
    try:
        path = resolve_ingest_path(service.settings.data_dir, body.path)
    except IngestPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="Ingest path does not exist")
    stats = service.ingest_path(path)
    return IngestResponse(
        files=stats.files,
        chunks=stats.chunks,
        flagged_chunks=stats.flagged_chunks,
        sources=stats.sources,
    )


@router.post("/ingest/upload", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def upload(
    service: KnowledgeService = Depends(get_service),
    files: list[UploadFile] = File(...),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one file")
    prepared: list[tuple[str, bytes]] = []
    for upload_file in files:
        name = Path(upload_file.filename or "upload.md").name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or name}")
        raw = await upload_file.read()
        if len(raw) > service.settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"{name} exceeds the upload limit")
        prepared.append((name, raw))
    upload_dir = service.settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, raw in prepared:
        target = upload_dir / name
        target.write_bytes(raw)
        written.append(target)
    files_count = 0
    chunks = 0
    flagged = 0
    sources: list[str] = []
    for path in written:
        stats = service.ingest_path(path)
        files_count += stats.files
        chunks += stats.chunks
        flagged += stats.flagged_chunks
        sources.extend(stats.sources)
    return IngestResponse(
        files=files_count,
        chunks=chunks,
        flagged_chunks=flagged,
        sources=sorted(set(sources)),
    )


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(body: QueryRequest, service: KnowledgeService = Depends(get_service)) -> QueryResponse:
    try:
        result = service.query(body.question, top_k=body.top_k)
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        retrieved=result.retrieved,
        quarantined=result.quarantined,
    )


@router.post("/evaluate", response_model=EvaluationResponse, dependencies=[Depends(require_api_key)])
def evaluate(body: EvaluateRequest, service: KnowledgeService = Depends(get_service)) -> EvaluationResponse:
    report = service.evaluate(
        question=body.question,
        answer=body.answer,
        contexts=body.contexts,
        ground_truth=body.ground_truth,
        use_llm_judge=body.use_llm_judge,
    )
    return EvaluationResponse(**report.model_dump())

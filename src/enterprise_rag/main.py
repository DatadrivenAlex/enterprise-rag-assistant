"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from enterprise_rag import __version__
from enterprise_rag.config import Settings, get_settings
from enterprise_rag.document_loader import IngestPathError, resolve_ingest_path
from enterprise_rag.rag_pipeline import KnowledgeService
from enterprise_rag.routes import router
from enterprise_rag.schemas import HealthResponse

logger = logging.getLogger("enterprise_rag")


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active: Settings = app.state.settings
        configure_logging(active)
        service = KnowledgeService(active)
        app.state.service = service
        if active.auto_ingest_dir:
            _auto_ingest(service, active)
        yield

    app = FastAPI(
        title=configured.app_name,
        version=__version__,
        summary="Enterprise retrieval-augmented generation with citations and guardrails.",
        description=(
            "Ask questions over ingested company documents. Answers cite the "
            "passages they came from. Prompt-injection attempts in the question "
            "or in retrieved text are refused or quarantined. The default "
            "configuration runs offline."
        ),
        lifespan=lifespan,
    )
    app.state.settings = configured
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configured.cors_origin_list or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service=configured.app_name, version=__version__)

    @app.get("/ready", response_model=HealthResponse, tags=["ops"])
    def ready() -> HealthResponse:
        service: KnowledgeService | None = getattr(app.state, "service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="starting")
        service.store.count()
        return HealthResponse(status="ready", service=configured.app_name, version=__version__)

    return app


def _auto_ingest(service: KnowledgeService, settings: Settings) -> None:
    assert settings.auto_ingest_dir is not None
    try:
        path = resolve_ingest_path(settings.data_dir, settings.auto_ingest_dir)
    except IngestPathError:
        logger.warning("auto ingest path is outside the data directory")
        return
    if not path.exists():
        logger.warning("auto ingest path does not exist: %s", path)
        return
    stats = service.ingest_path(path)
    logger.info("auto-ingested %s chunks from %s files", stats.chunks, stats.files)


app = create_app()


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "enterprise_rag.main:app",
        host=settings.host,
        port=settings.port,
        factory=False,
    )


if __name__ == "__main__":
    main()

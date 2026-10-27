"""HTTP request and response models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "How many paid vacation days do employees receive?", "top_k": 4}
            ]
        }
    )

    question: str = Field(min_length=1, max_length=20000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    marker: int
    source: str
    chunk_id: str
    score: float
    excerpt: str
    title: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool = False
    refusal_reason: str | None = None
    retrieved: int = 0
    quarantined: int = 0


class IngestRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1024)


class IngestResponse(BaseModel):
    files: int
    chunks: int
    flagged_chunks: int
    sources: list[str]


class EvaluateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    answer: str = Field(min_length=1, max_length=20000)
    contexts: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    ground_truth: str | None = None
    use_llm_judge: bool = False


class EvaluationResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    citation_support: float
    context_recall: float | None = None
    answer_correctness: float | None = None
    overall: float
    mode: str
    reasoning: str


class SourceListResponse(BaseModel):
    sources: list[str]
    chunks: int


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

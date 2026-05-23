from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class ChunkSource(BaseModel):
    filename: str
    page: int
    score: float
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[ChunkSource]
    retrieved_chunks: List[ChunkSource]
    latency_ms: int


class IngestResponse(BaseModel):
    pdfs_processed: int
    chunks_created: int
    chunks_stored: int
    skipped_pdfs: List[str]
    duration_seconds: float


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int

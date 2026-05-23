from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import time
import os

from app.models import QueryRequest, QueryResponse, IngestResponse, HealthResponse
from app.pipeline import RAGPipeline
from app.config import settings

pipeline: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = RAGPipeline()
    yield


app = FastAPI(
    title="RAG Chatbot — NavGurukul Hackathon",
    description="Retrieval-Augmented Generation over large PDF corpus",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=FileResponse)
def root():
    index = os.path.join(static_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "RAG Chatbot API — visit /docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    doc_count = pipeline.vector_store.count() if pipeline else 0
    return HealthResponse(status="ok", chunks_indexed=doc_count)


@app.post("/ingest", response_model=IngestResponse)
def ingest(pdf_dir: str = None):
    """Ingest all PDFs from data/pdfs or a specified directory."""
    try:
        target = pdf_dir or settings.PDF_DIR
        result = pipeline.ingest(target)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Answer a user question using RAG pipeline."""
    try:
        t0 = time.perf_counter()
        result = pipeline.query(req.question, top_k=req.top_k)
        latency_ms = round((time.perf_counter() - t0) * 1000)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            retrieved_chunks=result["retrieved_chunks"],
            latency_ms=latency_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

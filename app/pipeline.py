import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

from app.config import settings
from app.ingestion import extract_text_from_pdf, chunk_pages
from app.vector_store import VectorStore
from app.reranker import rerank
from app.llm import call_llm

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline:
      PDF → extract → chunk → embed → store → retrieve → rerank → LLM → answer
    """

    def __init__(self):
        self.vector_store = VectorStore()

    # ─── Ingestion ────────────────────────────────────────────────────────────

    def ingest(self, pdf_dir: str) -> Dict[str, Any]:
        """
        Ingest all PDFs from a directory.
        Skips PDFs already in the vector store (idempotent).
        """
        t0 = time.perf_counter()
        pdf_paths = list(Path(pdf_dir).glob("*.pdf"))

        if not pdf_paths:
            raise FileNotFoundError(f"No PDFs found in: {pdf_dir}")

        total_chunks = 0
        processed = 0
        skipped = []

        for pdf_path in pdf_paths:
            filename = pdf_path.name

            if self.vector_store.already_ingested(filename):
                logger.info(f"Skipping already-ingested: {filename}")
                skipped.append(filename)
                continue

            logger.info(f"Processing: {filename}")
            pages = extract_text_from_pdf(str(pdf_path), ocr_enabled=settings.OCR_ENABLED)

            if not pages:
                logger.warning(f"No text extracted from {filename}")
                skipped.append(filename)
                continue

            chunks = chunk_pages(
                pages,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP,
            )

            stored = self.vector_store.add_chunks(chunks)
            total_chunks += stored
            processed += 1
            logger.info(f"  → {len(pages)} pages, {stored} chunks stored")

        duration = round(time.perf_counter() - t0, 2)
        return {
            "pdfs_processed": processed,
            "chunks_created": total_chunks,
            "chunks_stored": self.vector_store.count(),
            "skipped_pdfs": skipped,
            "duration_seconds": duration,
        }

    # ─── Query ────────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Full RAG query:
          1. Retrieve top candidate chunks via vector search
          2. Rerank candidates
          3. Generate answer with LLM conditioned on top chunks
          4. Return answer + source citations
        """
        top_k = top_k or settings.TOP_K

        # Step 1: Retrieve more candidates than needed, then rerank
        candidates = self.vector_store.search(question, top_k=top_k * 3)

        if not candidates:
            return {
                "answer": "No documents have been ingested yet. Please run /ingest first.",
                "sources": [],
                "retrieved_chunks": [],
            }

        # Step 2: Rerank
        top_chunks = rerank(question, candidates, top_k=top_k)

        # Step 3: Generate
        answer = call_llm(question, top_chunks)

        # Step 4: Build source citations (deduplicated)
        seen = set()
        sources = []
        for chunk in top_chunks:
            key = (chunk["source"], chunk["page"])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "filename": chunk["source"],
                    "page": chunk["page"],
                    "score": chunk.get("rerank_score", chunk.get("score", 0.0)),
                    "text_preview": chunk["text"][:200] + "...",
                })

        retrieved_chunks = [
            {
                "filename": c["source"],
                "page": c["page"],
                "score": round(c.get("rerank_score", c.get("score", 0.0)), 4),
                "text_preview": c["text"][:300] + "...",
            }
            for c in top_chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
        }

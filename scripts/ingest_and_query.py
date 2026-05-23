#!/usr/bin/env python3
"""
Standalone script to ingest PDFs and optionally run a test query.
Usage:
    python scripts/ingest_and_query.py --pdf-dir ./data/pdfs
    python scripts/ingest_and_query.py --pdf-dir ./data/pdfs --query "What is retrieval augmented generation?"
"""
import argparse
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline import RAGPipeline
from app.config import settings


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    parser.add_argument("--pdf-dir", default=settings.PDF_DIR, help="Directory containing PDFs")
    parser.add_argument("--query", default=None, help="Optional test query after ingestion")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  RAG Chatbot — Ingestion + Query CLI")
    print(f"{'='*60}\n")

    pipeline = RAGPipeline()

    # Ingestion
    print(f"[1/2] Ingesting PDFs from: {args.pdf_dir}")
    t0 = time.perf_counter()
    try:
        result = pipeline.ingest(args.pdf_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  ✓ PDFs processed : {result['pdfs_processed']}")
    print(f"  ✓ Chunks stored  : {result['chunks_stored']}")
    print(f"  ✓ Skipped        : {result['skipped_pdfs']}")
    print(f"  ✓ Duration       : {result['duration_seconds']}s")

    # Query
    if args.query:
        print(f"\n[2/2] Running query: \"{args.query}\"")
        t1 = time.perf_counter()
        res = pipeline.query(args.query, top_k=args.top_k)
        latency = round((time.perf_counter() - t1) * 1000)

        print(f"\n{'─'*60}")
        print(f"ANSWER ({latency}ms):\n")
        print(res["answer"])
        print(f"\n{'─'*60}")
        print("SOURCES:")
        for s in res["sources"]:
            print(f"  • {s['source']} — page {s['page']}  (score: {s['score']})")
        print(f"\nTOP CHUNKS:")
        for i, c in enumerate(res["retrieved_chunks"], 1):
            print(f"\n  [{i}] {c['source']} p.{c['page']} (score={c['score']})")
            print(f"      {c['text'][:200]}...")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()

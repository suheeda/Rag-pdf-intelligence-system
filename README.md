<div align="center">

# 🔍 RAG PDF Intelligence System

### Retrieval-Augmented Generation over large private PDF corpora

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6B35?style=flat-square)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

**Open-source stack · sub-5s latency · full source citations · zero paid APIs required**

[Live Demo](#setup--run) · [API Docs](http://localhost:8000/docs) · [Architecture](#architecture)

</div>

---

## What It Does

Ask questions in plain English against a private corpus of PDFs — the system retrieves the most relevant passages, generates a grounded answer, and cites the exact document and page number. No hallucination from thin air; every answer traces back to a source.

| Feature | Detail |
|---|---|
| PDF ingestion | Native text (PyMuPDF) + OCR fallback (Tesseract) |
| Embeddings | `all-MiniLM-L6-v2` — free, local, no API key needed |
| Vector DB | ChromaDB with HNSW index — persistent, open-source |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) for precision |
| LLM | Groq (free) · Gemini · Ollama (fully offline) |
| Latency | ~2–5s end-to-end on typical hardware |

---

## Screenshots

### Chat Interface — Ready State

> The UI loads at `localhost:8000`. Corpus stats update live; click **Ingest PDFs** to index your documents.

![RAG Chatbot UI — Ready](https://raw.githubusercontent.com/suheeda/Rag-pdf-intelligence-system/main/docs/screenshots/ui_ready.png)

---

### Ingestion Complete

> After ingestion, the sidebar confirms PDFs processed, chunks stored, and time taken. Re-ingestion is idempotent — already-indexed files are skipped automatically.

![Ingestion Complete](https://raw.githubusercontent.com/suheeda/Rag-pdf-intelligence-system/main/docs/screenshots/ingestion_complete.png)

---

### RAG Response with Source Citations

> The chatbot generates a grounded answer and lists the exact source documents and page numbers used. The sidebar shows retrieved chunks with relevance scores.

![RAG Response with Citations](https://raw.githubusercontent.com/suheeda/Rag-pdf-intelligence-system/main/docs/screenshots/rag_response_citations.png)

---

### Retrieval Visualization — Top Chunks

> Each answer surfaces the top retrieved chunks in the sidebar — showing filename, page number, relevance score, and a text preview for full transparency.

![Retrieved Chunks Panel](https://raw.githubusercontent.com/suheeda/Rag-pdf-intelligence-system/main/docs/screenshots/retrieved_chunks.png)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                        │
│                                                             │
│  User Query                                                 │
│      │                                                      │
│      ▼                                                      │
│  FastAPI /query                                             │
│      │                                                      │
│      ├──► Bi-Encoder Embedding (all-MiniLM-L6-v2)          │
│      │         └── ChromaDB HNSW Search → top-15 chunks    │
│      │                                                      │
│      ├──► Cross-Encoder Reranking (ms-marco-MiniLM)        │
│      │         └── Reranked top-5 chunks                   │
│      │                                                      │
│      └──► LLM Generation (Groq / Gemini / Ollama)          │
│                └── Answer + Source Citations                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                      │
│                                                             │
│  PDF Files                                                  │
│      │                                                      │
│      ├──► PyMuPDF  →  Native text extraction               │
│      ├──► Tesseract →  OCR fallback (scanned pages)        │
│      ├──► Cleaner   →  Remove headers/footers, normalize   │
│      ├──► Chunker   →  800 tokens, 150 overlap             │
│      └──► ChromaDB  →  Embedded + persisted to disk        │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| PDF Extraction | PyMuPDF + Tesseract | Fast native text + OCR for scanned pages |
| Embeddings | `all-MiniLM-L6-v2` | Free, local, strong semantic quality |
| Vector DB | ChromaDB (HNSW) | Zero-config, persistent, fully open-source |
| Reranker | `ms-marco-MiniLM-L-6-v2` | ~15% precision gain at low latency cost |
| LLM | Groq `llama3-8b-8192` (primary) | Free tier, ~500ms generation, no quota issues |
| API | FastAPI | Typed, async, auto-generates `/docs` |
| Frontend | Vanilla HTML/JS | Zero dependencies, instant cold start |

---

## Project Structure

```
rag_chatbot/
├── app/
│   ├── main.py            # FastAPI app + route handlers
│   ├── pipeline.py        # RAG orchestrator (ingest + query)
│   ├── ingestion.py       # PDF extraction, OCR, chunking
│   ├── vector_store.py    # ChromaDB wrapper + embeddings
│   ├── reranker.py        # Cross-encoder reranking
│   ├── llm.py             # LLM clients: Groq, Gemini, Ollama
│   ├── models.py          # Pydantic request/response schemas
│   └── config.py          # Settings loaded from .env
├── static/
│   └── index.html         # Chat UI (served at /)
├── data/pdfs/             # Drop your PDFs here
├── scripts/
│   ├── ingest_and_query.py    # CLI: ingest + test query
│   └── evaluate.py            # Latency & citation evaluation
├── tests/
│   └── test_pipeline.py       # Pytest unit tests
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Clone & install

```bash
git clone https://github.com/suheeda/Rag-pdf-intelligence-system
cd Rag-pdf-intelligence-system/rag_chatbot

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` — the recommended free setup:

```env
# Groq: free, no credit card — get key at https://console.groq.com
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant```

> **Alternatives:** Set `LLM_PROVIDER=gemini` + `GEMINI_API_KEY`, or `LLM_PROVIDER=ollama` for fully offline inference (run `ollama pull mistral` first).

### 3. Add PDFs

```bash
# Place your PDF files in:
data/pdfs/
```

### 4. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)** — the chat UI loads automatically.

### 5. Ingest & query

Click **"Ingest PDFs"** in the UI sidebar, or use the CLI:

```bash
python scripts/ingest_and_query.py \
  --pdf-dir ./data/pdfs \
  --query "What are the key recommendations?"
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System status + total chunks indexed |
| `POST` | `/ingest` | Ingest all PDFs from `data/pdfs/` |
| `POST` | `/query` | RAG query → answer + citations + chunks |
| `GET` | `/docs` | Interactive Swagger UI |

**Example query:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "top_k": 5}'
```

**Response:**
```json
{
  "answer": "The key findings include... \n\nSources:\ndocument.pdf, Pages 12-13",
  "sources": [
    {"filename": "document.pdf", "page": 12, "score": 0.91, "text_preview": "..."}
  ],
  "retrieved_chunks": [...],
  "latency_ms": 1996
}
```

---

## Technical Decisions

### 1. ChromaDB over Pinecone / Weaviate
ChromaDB runs fully locally, persists to disk, and uses HNSW internally — equivalent ANN performance to managed cloud solutions for corpora under 1M chunks. Satisfies the open-source requirement with zero infrastructure overhead.

**Trade-off:** No horizontal scaling or multi-node support. Mitigatable by switching to Qdrant (drop-in compatible) at scale.

### 2. Two-stage retrieval: bi-encoder + cross-encoder reranker
The bi-encoder retrieves top-15 candidates in ~20ms via ANN. A cross-encoder then re-scores all 15 against the exact query string, achieving higher precision than bi-encoder alone at a cost of ~200–400ms. Net result: noticeably better answer relevance with total latency still well within the 5s target.

**Trade-off:** Disable with `RERANK_ENABLED=false` if hardware is constrained.

### 3. PyMuPDF primary + Tesseract OCR fallback
PyMuPDF extracts native text orders of magnitude faster than OCR. Pages returning fewer than 50 characters trigger a Tesseract pass at 200 DPI — covering scanned or image-heavy PDFs without penalising text-native documents.

---

## Latency Profile

| Stage | Typical |
|---|---|
| Query embedding | 20–50ms |
| Vector search (HNSW) | 10–30ms |
| Cross-encoder reranking | 150–400ms |
| LLM generation (Groq) | 400–1500ms |
| **End-to-end** | **~1–3s** |

> First query adds ~1s for model warm-up. All subsequent queries are within target range.

---

## Evaluation

```bash
python scripts/evaluate.py --questions \
  "What is the main objective of this document?" \
  "What are the key findings?" \
  "What recommendations are made?"
```

Outputs p50/p95 latency, citation accuracy, and per-question results to `eval_results.json`.

---

## Roadmap

- [ ] Streaming LLM responses (SSE) for perceived speed improvement
- [ ] Hybrid search: BM25 lexical + dense vector with RRF fusion
- [ ] Metadata filters: query by PDF name, date range, or section
- [ ] Golden Q&A evaluation harness per document
- [ ] Docker Compose for single-command setup

---

## Author

**Suheeda SF** — AI/ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-suheeda-181717?style=flat-square&logo=github)](https://github.com/suheeda)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-2563eb?style=flat-square)](https://portfolio-project-alpha-mocha.vercel.app/)
[![Email](https://img.shields.io/badge/Email-suheedasf10@gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:suheedasf10@gmail.com)

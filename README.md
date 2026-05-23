# RAG Chatbot — NavGurukul AI/ML Hackathon

> Retrieval-Augmented Generation over large private PDF corpora.  
> Open-source stack · sub-5s latency · full source citations.

---

## Problem Statement

**Challenge 1: RAG Chatbot**  
Build a RAG chatbot that answers user queries from a large private PDF corpus (≥10 PDFs, ≥200 pages each) using free/open-source embedding models and vector DB, with end-to-end response latency of 2–5 seconds.

---

## Solution Architecture

```
User Query
    │
    ▼
FastAPI /query
    │
    ├─► Vector Search (ChromaDB HNSW)
    │       └── all-MiniLM-L6-v2 embeddings
    │
    ├─► Cross-Encoder Reranking
    │       └── ms-marco-MiniLM-L-6-v2
    │
    └─► LLM Generation (Gemini 1.5 Flash / Ollama)
            └── Answer + Source Citations

PDF Ingestion Pipeline:
PDF → PyMuPDF (native) / Tesseract (OCR) → Chunking (800 tokens, 150 overlap)
    → Sentence Transformers → ChromaDB (persistent HNSW index)
```

### Tech Stack

| Component        | Choice                              | Why                                      |
|------------------|-------------------------------------|------------------------------------------|
| PDF Extraction   | PyMuPDF + Tesseract                 | Fast native + OCR fallback               |
| Embeddings       | all-MiniLM-L6-v2 (HuggingFace)     | Free, fast (768d → 384d), strong quality |
| Vector DB        | ChromaDB                            | Free, local, HNSW index built-in         |
| Reranker         | ms-marco-MiniLM cross-encoder       | +15% relevance with minimal latency cost |
| LLM              | Gemini 1.5 Flash (or Ollama)        | Free tier / fully offline option         |
| API              | FastAPI                             | Fast, typed, auto-docs                   |
| Frontend         | Vanilla HTML/JS                     | Zero dependencies, instant load          |

---

## Setup & Run

### 1. Clone & install

```bash
git clone https://github.com/suheeda/rag-chatbot
cd rag-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env: add your GEMINI_API_KEY (or set LLM_PROVIDER=ollama)
```

### 3. Add PDFs

```bash
# Place your PDFs in:
data/pdfs/your_document.pdf
```

### 4. Start server

```bash
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** — the chat UI loads automatically.

### 5. Ingest PDFs

Click **"Ingest PDFs"** in the UI, or via CLI:

```bash
python scripts/ingest_and_query.py --pdf-dir ./data/pdfs
```

### 6. Ask questions

Via UI, or API:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key recommendations in the report?"}'
```

---

## API Endpoints

| Method | Endpoint  | Description                          |
|--------|-----------|--------------------------------------|
| GET    | `/health` | System status + chunk count          |
| POST   | `/ingest` | Ingest PDFs from `data/pdfs/`        |
| POST   | `/query`  | RAG query → answer + citations       |
| GET    | `/docs`   | Interactive Swagger API docs         |

---

## Key Technical Decisions

### 1. ChromaDB over Pinecone / Weaviate
ChromaDB runs fully locally with zero configuration, persists to disk, and uses HNSW internally — identical performance characteristics to managed solutions for corpora under 1M chunks. This satisfies the open-source requirement without infra overhead.

**Trade-off:** No horizontal scaling or multi-node support. Acceptable for hackathon scope.

### 2. Two-stage retrieval: bi-encoder + cross-encoder rerank
Bi-encoder (sentence-transformers) retrieves top-15 candidates fast via ANN. A cross-encoder then re-scores against the exact query for precision. This pattern consistently outperforms single-stage retrieval in relevance without proportional latency cost.

**Trade-off:** ~200–400ms added latency for reranking. Disable with `RERANK_ENABLED=false` if hitting latency targets.

### 3. PyMuPDF + Tesseract OCR fallback
PyMuPDF extracts native text at native speed. Pages with fewer than 50 characters of extracted text trigger Tesseract OCR at 200 DPI — covering scanned documents without unnecessary overhead on text-native PDFs.

---

## Latency Profile (typical hardware)

| Stage              | Typical (ms) |
|--------------------|-------------|
| Query embedding    | 20–50ms      |
| Vector search      | 10–30ms      |
| Reranking (top 15) | 150–400ms    |
| LLM generation     | 800–2500ms   |
| **Total**          | **~1–3s**    |

> Note: First query is slower (~+1s) due to model warm-up. Subsequent queries hit the target range.

---

## Evaluation

```bash
python scripts/evaluate.py --questions \
  "What is the main objective?" \
  "What are the key findings?" \
  "What recommendations are made?"
```

Outputs: p50/p95 latency, citation accuracy rate, per-question results → `eval_results.json`

---

## Scaling Considerations

| Bottleneck            | Mitigation                                              |
|-----------------------|---------------------------------------------------------|
| Ingestion speed       | Parallel PDF processing (multiprocessing.Pool)          |
| RAM for large corpora | Swap ChromaDB for Qdrant with quantized HNSW            |
| LLM latency           | Switch to Ollama (local) or use streaming responses     |
| Reranker latency      | Set `RERANK_ENABLED=false` or batch cross-encoder calls |

---

## What I'd Improve With More Time

- Streaming LLM responses to the frontend (SSE) for perceived speed
- Hybrid search: BM25 lexical + dense vector with RRF fusion
- Metadata filters: query only specific PDFs or date ranges
- Evaluation harness with golden Q&A pairs per document
- Docker Compose for one-command setup

---

## Project Structure

```
rag_chatbot/
├── app/
│   ├── main.py          # FastAPI app + route handlers
│   ├── pipeline.py      # RAG orchestrator (ingest + query)
│   ├── ingestion.py     # PDF extraction, OCR, chunking
│   ├── vector_store.py  # ChromaDB wrapper + embedding
│   ├── reranker.py      # Cross-encoder reranking
│   ├── llm.py           # LLM clients (Gemini, Ollama, OpenAI)
│   ├── models.py        # Pydantic schemas
│   └── config.py        # Settings from .env
├── static/
│   └── index.html       # Chat UI
├── data/pdfs/           # Place your PDFs here
├── scripts/
│   ├── ingest_and_query.py   # CLI ingestion + test query
│   └── evaluate.py           # Latency + citation evaluation
├── tests/
│   └── test_pipeline.py      # Pytest unit tests
├── .env.example
├── requirements.txt
└── README.md
```

---

## Author

**Suheeda SF**  
suheedasf10@gmail.com | [GitHub](https://github.com/suheeda) | [Portfolio](https://portfolio-project-alpha-mocha.vercel.app/)

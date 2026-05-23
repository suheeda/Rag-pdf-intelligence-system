import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "chunks_indexed" in data


def test_query_no_docs():
    """Should return graceful message when no docs are ingested."""
    with patch("app.pipeline.RAGPipeline.query") as mock_query:
        mock_query.return_value = {
            "answer": "No documents have been ingested yet.",
            "sources": [],
            "retrieved_chunks": [],
        }
        resp = client.post("/query", json={"question": "What is machine learning?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data


def test_query_validation():
    """Short question should fail validation."""
    resp = client.post("/query", json={"question": "hi"})
    assert resp.status_code == 422


def test_chunking_logic():
    from app.ingestion import chunk_pages
    fake_pages = [{"text": " ".join(["word"] * 1000), "source": "test.pdf", "page": 1, "pdf_path": "/tmp/test.pdf"}]
    chunks = chunk_pages(fake_pages, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert "text" in c
        assert "source" in c
        assert "page" in c


def test_clean_text():
    from app.ingestion import _clean_text
    dirty = "Title\n\n\n\n123\n\nSome   actual   content here."
    clean = _clean_text(dirty)
    assert "123" not in clean
    assert "actual content" in clean

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-backed vector store with sentence-transformer embeddings.
    Uses HNSW index (ChromaDB default) for fast ANN search.
    """

    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore ready — {self.collection.count()} chunks in '{settings.COLLECTION_NAME}'"
        )

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed and store chunks. Returns number of chunks added."""
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c["source"],
                "page": c["page"],
                "chunk_index": c.get("chunk_index", 0),
            }
            for c in chunks
        ]
        ids = [
            f"{c['source']}__p{c['page']}__c{c.get('chunk_index', 0)}"
            for c in chunks
        ]

        # Batch embed for throughput
        embeddings = self.embedding_model.encode(
            texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
        ).tolist()

        # Upsert to avoid duplicates on re-ingest
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Stored {len(chunks)} chunks → collection now has {self.count()} total")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed query and retrieve top-k nearest chunks with metadata + scores.
        """
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "score": round(1 - dist, 4),  # cosine distance → similarity
            })

        return hits

    def already_ingested(self, filename: str) -> bool:
        """Check if a PDF has already been embedded (fast existence check)."""
        results = self.collection.get(
            where={"source": filename}, limit=1, include=[]
        )
        return len(results["ids"]) > 0

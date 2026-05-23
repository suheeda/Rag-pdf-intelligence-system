import logging
from typing import List, Dict
from app.config import settings

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANKER_MODEL)
            logger.info(f"Reranker loaded: {RERANKER_MODEL}")
        except Exception as e:
            logger.warning(f"Could not load reranker ({e}). Skipping rerank.")
    return _reranker


def rerank(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Re-score candidate chunks with a cross-encoder.
    Falls back to original bi-encoder order if reranker unavailable.
    """
    if not settings.RERANK_ENABLED or not chunks:
        return chunks[:top_k]

    reranker = get_reranker()
    if reranker is None:
        return chunks[:top_k]

    try:
        pairs = [(query, c["text"]) for c in chunks]
        scores = reranker.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return reranked[:top_k]
    except Exception as e:
        logger.warning(f"Reranking failed ({e}); using original order")
        return chunks[:top_k]

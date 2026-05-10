"""Lazy-loaded sentence-transformers wrapper for Recall hybrid search.

Falls back gracefully to None when sentence-transformers is not installed,
allowing BM25-only mode for dev environments without the embedding dependency.

Model: BAAI/bge-small-en-v1.5 — 33M params, <100ms/query on CPU, strong retrieval quality.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_MODEL_NAME)
            logger.info("embeddings_model_loaded", extra={"model": _MODEL_NAME})
        except ImportError:
            logger.warning("sentence_transformers_not_installed — running in BM25-only mode")
            _model = None
    return _model


def embed(texts: list[str]) -> np.ndarray | None:
    """Encode texts into L2-normalized embeddings. Returns None if model unavailable."""
    if not texts:
        return None
    model = get_model()
    if model is None:
        return None
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(query: str) -> np.ndarray | None:
    """Encode a single query string."""
    result = embed([query])
    if result is None:
        return None
    return result[0]


def cosine_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Dot product of L2-normalized vectors = cosine similarity."""
    return doc_vecs @ query_vec


def vec_to_blob(vec: np.ndarray) -> bytes:
    """Serialize float32 ndarray to bytes for SQLite BLOB storage."""
    return vec.astype(np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    """Deserialize bytes from SQLite BLOB back to float32 ndarray."""
    return np.frombuffer(blob, dtype=np.float32)

"""
Stage 4 of Ingestion & Stage 2 of Querying: Local Embedding Generation.
Uses Sentence Transformers (`all-MiniLM-L6-v2`) to convert text into
384-dimensional dense vectors that capture semantic meaning.
"""

import time
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import config

_MODEL_INSTANCE: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model locally."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        try:
            _MODEL_INSTANCE = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{config.EMBEDDING_MODEL_NAME}': {exc}"
            ) from exc
    return _MODEL_INSTANCE


def generate_embeddings(texts: List[str]) -> Tuple[List[List[float]], int, float]:
    """
    Generate normalized embeddings for a list of strings.

    Returns:
        (vectors_as_lists, embedding_dimension, elapsed_seconds)
    """
    if not texts:
        return [], 0, 0.0

    model = get_embedding_model()
    start = time.perf_counter()
    try:
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception as exc:
        raise RuntimeError(f"Embedding generation failed: {exc}") from exc
    elapsed = time.perf_counter() - start

    vector_lists = [v.tolist() for v in vectors]
    dim = len(vector_lists[0]) if vector_lists else 0
    return vector_lists, dim, elapsed


def embed_query(query: str) -> Tuple[List[float], int, float]:
    """Generate an embedding vector for a single user question."""
    vectors, dim, elapsed = generate_embeddings([query])
    return vectors[0], dim, elapsed


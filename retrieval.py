"""
Stages 3 & 4 of Querying: Semantic Retrieval & Context Selection.
Converts the query to an embedding, searches ChromaDB, applies an optional
similarity threshold, and identifies matching sentences for educational display.
"""

import re
from typing import Dict, Any, List
import config
from rag import embeddings, vector_store


def retrieve_relevant_chunks(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD,
    collection_name: str = config.CHROMA_COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    Execute Stages 2, 3, and 4 of the RAG query pipeline:
      - Generate query embedding
      - Run ChromaDB similarity search
      - Filter by similarity threshold (if > 0)
      - Extract relevant sentences for educational highlighting
    """
    query_vec, dim, embed_time = embeddings.embed_query(query)
    search_res = vector_store.query_similar(
        query_embedding=query_vec,
        top_k=top_k,
        collection_name=collection_name,
    )

    all_candidates = search_res["results"]
    total_searched = search_res["total_searched"]

    # Apply similarity threshold if configured
    selected = [
        item for item in all_candidates if item["similarity"] >= similarity_threshold
    ]

    # Annotate each selected chunk with matching sentences for Stage 4 visual inspection
    query_terms = _extract_keywords(query)
    for item in selected:
        item["highlighted_sentences"] = _find_relevant_sentences(
            item["text"], query_terms
        )

    return {
        "query": query,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embedding_dim": dim,
        "embed_time_sec": round(embed_time, 4),
        "query_vector": query_vec,
        "total_chunks_searched": total_searched,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "similarity_metric": "Cosine Similarity (1 - Chroma Cosine Distance)",
        "all_candidates": all_candidates,
        "selected_chunks": selected,
    }


def _extract_keywords(query: str) -> List[str]:
    """Extract meaningful lowercase keywords from the user query."""
    stop_words = {
        "what", "is", "the", "a", "an", "are", "how", "do", "does", "if",
        "can", "to", "for", "of", "in", "on", "and", "or", "when", "where",
        "why", "who", "which", "with", "about", "student", "students",
    }
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return [w for w in words if len(w) > 2 and w not in stop_words]


def _find_relevant_sentences(text: str, keywords: List[str]) -> List[str]:
    """Return sentences from a chunk that overlap with query keywords."""
    if not keywords:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    matches = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_lower = s_clean.lower()
        if any(kw in s_lower for kw in keywords):
            matches.append(s_clean)
    return matches[:3]


"""
Stage 5 of Ingestion & Stage 3 of Querying: ChromaDB Vector Store.
Stores document chunks, metadata, and embedding vectors persistently on disk
and performs cosine-similarity vector search.
"""

from typing import List, Dict, Any, Optional
import chromadb
import config


_CLIENT: Any = None


def get_chroma_client() -> Any:
    """Return a cached ChromaDB PersistentClient."""
    global _CLIENT
    if _CLIENT is None:
        try:
            _CLIENT = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        except Exception as exc:
            raise RuntimeError(f"ChromaDB connection failed: {exc}") from exc
    return _CLIENT


def get_collection(collection_name: str = config.CHROMA_COLLECTION_NAME):
    """Get or create the ChromaDB collection configured with cosine distance."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    collection_name: str = config.CHROMA_COLLECTION_NAME,
) -> int:
    """Store chunks, their embeddings, and metadata in ChromaDB."""
    if not chunks:
        return 0
    collection = get_collection(collection_name)
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


def delete_by_document_id(
    document_id: str,
    collection_name: str = config.CHROMA_COLLECTION_NAME,
) -> None:
    """Delete all vectors belonging to a specific document_id."""
    collection = get_collection(collection_name)
    collection.delete(where={"document_id": document_id})


def get_chunks_by_document_id(
    document_id: str,
    collection_name: str = config.CHROMA_COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """Fetch all stored chunks and metadata for a given document_id."""
    collection = get_collection(collection_name)
    res = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )
    items = []
    for cid, doc, meta in zip(
        res.get("ids", []),
        res.get("documents", []),
        res.get("metadatas", []),
    ):
        items.append({"chunk_id": cid, "text": doc, "metadata": meta})
    items.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
    return items


def query_similar(
    query_embedding: List[float],
    top_k: int = config.DEFAULT_TOP_K,
    collection_name: str = config.CHROMA_COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    Search ChromaDB for the top-K most semantically similar chunks.
    Returns raw ChromaDB ids, documents, metadatas, and cosine distances.
    """
    collection = get_collection(collection_name)
    total = collection.count()
    if total == 0:
        return {"total_searched": 0, "results": []}

    k = min(max(1, top_k), total)
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]

    results = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        # For cosine space in ChromaDB, distance = 1 - cosine_similarity
        similarity = max(0.0, min(1.0, 1.0 - float(dist)))
        results.append(
            {
                "chunk_id": cid,
                "text": doc,
                "metadata": meta,
                "distance": round(float(dist), 4),
                "similarity": round(similarity, 4),
            }
        )

    return {"total_searched": total, "results": results}


def get_total_vectors(collection_name: str = config.CHROMA_COLLECTION_NAME) -> int:
    """Return total stored vector count."""
    try:
        return get_collection(collection_name).count()
    except Exception:
        return 0


def health_check() -> Dict[str, Any]:
    """Check whether ChromaDB is reachable and return collection stats."""
    try:
        count = get_collection().count()
        return {"ok": True, "status": "Connected", "vector_count": count}
    except Exception as exc:
        return {"ok": False, "status": f"Error: {exc}", "vector_count": 0}


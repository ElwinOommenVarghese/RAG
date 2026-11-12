"""
End-to-End RAG Pipeline Orchestrator.
Provides clean, beginner-traceable functions for:
  1. Document Ingestion (Extract -> Clean -> Chunk -> Embed -> Store)
  2. Document Reprocessing & Deletion
  3. 7-Stage RAG Query Execution (Query -> Embed -> Search -> Context -> Prompt -> LLM -> Answer)
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
import config
from rag import (
    document_loader,
    text_cleaner,
    chunking,
    embeddings,
    vector_store,
    retrieval,
    prompt_builder,
    generation,
)
from utils import helpers


ProgressCallback = Optional[Callable[[int, str], None]]


def ingest_document(
    file_name: str,
    file_bytes: bytes,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    progress_callback: ProgressCallback = None,
    allow_overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Execute the full 5-step document ingestion pipeline and return a processing report.
    """
    def _notify(step: int, msg: str):
        if progress_callback:
            progress_callback(step, msg)

    file_hash = helpers.compute_sha256(file_bytes)
    registry = helpers.load_registry()

    existing = next(
        (d for d in registry if d["file_hash"] == file_hash or d["file_name"] == file_name),
        None,
    )
    if existing and not allow_overwrite:
        raise ValueError(
            f"Duplicate document detected: '{file_name}' is already in the Knowledge Base. "
            "Use 'Reprocess' or delete the existing document first."
        )

    doc_id = existing["document_id"] if existing else f"doc_{uuid.uuid4().hex[:8]}"
    stored_path = config.UPLOAD_DIR / f"{doc_id}_{file_name}"
    stored_path.write_bytes(file_bytes)

    try:
        # Step 1: Extract text
        _notify(1, f"Step 1/5: Extracting text from '{file_name}'...")
        raw_text, page_count, file_type = document_loader.extract_text_from_file(stored_path)

        # Step 2: Clean text
        _notify(2, "Step 2/5: Cleaning and normalizing extracted text...")
        cleaned_text = text_cleaner.clean_text(raw_text)
        if not cleaned_text:
            raise ValueError(f"Document '{file_name}' has no usable text after cleaning.")

        # Step 3: Split into chunks
        _notify(3, f"Step 3/5: Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")
        chunks = chunking.split_into_chunks(
            text=cleaned_text,
            document_id=doc_id,
            source_name=file_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise ValueError(f"Failed to produce any chunks for '{file_name}'.")

        # Step 4: Generate embeddings
        _notify(4, f"Step 4/5: Generating local embeddings for {len(chunks)} chunks...")
        vectors, dim, embed_time = embeddings.generate_embeddings([c["text"] for c in chunks])

        # Step 5: Store in ChromaDB
        _notify(5, "Step 5/5: Storing vectors, text, and metadata in ChromaDB...")
        if existing:
            vector_store.delete_by_document_id(doc_id)
        stored_count = vector_store.add_chunks(chunks, vectors)

    except Exception:
        # Clean up orphan file if ingestion failed for a brand-new doc
        if not existing and stored_path.exists():
            stored_path.unlink(missing_ok=True)
        raise

    report = {
        "document_id": doc_id,
        "file_name": file_name,
        "file_type": file_type,
        "file_size_bytes": len(file_bytes),
        "file_size_label": helpers.format_file_size(len(file_bytes)),
        "file_hash": file_hash,
        "stored_path": str(stored_path),
        "page_count": page_count,
        "extracted_text_length": len(cleaned_text),
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embedding_dim": dim,
        "embed_time_sec": round(embed_time, 3),
        "vectors_stored": stored_count,
        "status": "Indexed",
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    registry = [d for d in registry if d["document_id"] != doc_id]
    registry.append(report)
    helpers.save_registry(registry)
    helpers.append_activity(
        "upload",
        f"Ingested '{file_name}' ({page_count} pages, {len(chunks)} chunks, {stored_count} vectors)",
    )
    return report


def reprocess_document(
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """Re-chunk, re-embed, and re-store an existing document with updated settings."""
    registry = helpers.load_registry()
    doc = next((d for d in registry if d["document_id"] == document_id), None)
    if not doc:
        raise ValueError("Document not found in registry.")

    stored_path = Path(doc["stored_path"])
    if not stored_path.exists():
        raise ValueError(f"Original file for '{doc['file_name']}' is missing on disk.")

    return ingest_document(
        file_name=doc["file_name"],
        file_bytes=stored_path.read_bytes(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        progress_callback=progress_callback,
        allow_overwrite=True,
    )


def delete_document(document_id: str) -> str:
    """Remove a document from ChromaDB, disk uploads, and the registry."""
    registry = helpers.load_registry()
    doc = next((d for d in registry if d["document_id"] == document_id), None)
    if not doc:
        raise ValueError("Document not found.")

    vector_store.delete_by_document_id(document_id)

    stored_path = Path(doc.get("stored_path", ""))
    if stored_path.exists():
        stored_path.unlink(missing_ok=True)

    registry = [d for d in registry if d["document_id"] != document_id]
    helpers.save_registry(registry)
    helpers.append_activity("delete", f"Deleted '{doc['file_name']}' and its vectors from ChromaDB")
    return doc["file_name"]


def load_sample_documents(
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Ingest all sample fictional documents from `sample_documents/`."""
    reports = []
    for file_path in sorted(config.SAMPLE_DOCS_DIR.glob("*")):
        if file_path.suffix.lower() in document_loader.SUPPORTED_EXTENSIONS:
            try:
                report = ingest_document(
                    file_name=file_path.name,
                    file_bytes=file_path.read_bytes(),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    allow_overwrite=True,
                )
                reports.append(report)
            except ValueError:
                continue
    return reports


def run_rag_query(
    question: str,
    top_k: int = config.DEFAULT_TOP_K,
    similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD,
    chat_history: List[Dict[str, str]] | None = None,
    log_activity: bool = True,
) -> Dict[str, Any]:
    """
    Execute the complete 7-stage RAG pipeline and return a full trace object
    for both the Chatbot and the RAG Pipeline Explorer.
    """
    clean_q = question.strip()
    if not clean_q:
        raise ValueError("Please enter a non-empty question.")

    # Stages 2, 3, 4: Query Embedding -> Similarity Search -> Context Selection
    ret = retrieval.retrieve_relevant_chunks(
        query=clean_q,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    selected_chunks = ret["selected_chunks"]

    # Stage 5: Prompt Augmentation
    prompt_parts = prompt_builder.build_augmented_prompt(
        user_question=clean_q,
        selected_chunks=selected_chunks,
    )

    # Stage 6: LLM Generation
    gen = generation.generate_answer(
        system_instructions=prompt_parts["system_instructions"],
        llm_user_message=prompt_parts["llm_user_message"],
        has_context=len(selected_chunks) > 0,
        chat_history=chat_history,
    )

    # Stage 7: Final Answer + Source Citations
    sources = []
    seen_sources = set()
    for chunk in selected_chunks:
        src = chunk["metadata"].get("source", "Unknown")
        if src not in seen_sources:
            seen_sources.add(src)
            sources.append(src)

    if log_activity:
        helpers.append_activity(
            "question",
            f"Asked: \"{clean_q}\" (retrieved {len(selected_chunks)} chunks)",
        )

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage1_user_query": clean_q,
        "stage2_query_embedding": {
            "model": ret["embedding_model"],
            "dimension": ret["embedding_dim"],
            "embed_time_sec": ret["embed_time_sec"],
            "vector": ret["query_vector"],
        },
        "stage3_similarity_search": {
            "total_chunks_searched": ret["total_chunks_searched"],
            "top_k": ret["top_k"],
            "similarity_threshold": ret["similarity_threshold"],
            "metric": ret["similarity_metric"],
            "candidates": ret["all_candidates"],
        },
        "stage4_retrieved_context": selected_chunks,
        "stage5_prompt_augmentation": prompt_parts,
        "stage6_llm_generation": gen,
        "stage7_final_answer": {
            "answer": gen["answer"],
            "status": gen["status"],
            "error_message": gen["error_message"],
            "sources": sources,
        },
    }


"""
Stage 3 of Ingestion: Document Chunking.
Uses LangChain's RecursiveCharacterTextSplitter to split cleaned text into
overlapping chunks that preserve semantic context across boundaries.
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_into_chunks(
    text: str,
    document_id: str,
    source_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks with rich source metadata.

    Args:
        text: Cleaned document text.
        document_id: Unique ID for the parent document.
        source_name: Original filename (e.g., '01_student_attendance_policy.txt').
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        List of dicts containing chunk_id, text, char_count, and metadata.
    """
    if not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)
    chunks: List[Dict[str, Any]] = []

    for idx, chunk_text in enumerate(raw_chunks):
        chunk_id = f"{document_id}_chunk_{idx}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "metadata": {
                    "document_id": document_id,
                    "source": source_name,
                    "chunk_index": idx,
                    "total_chunks": len(raw_chunks),
                    "chunk_size_config": chunk_size,
                    "chunk_overlap_config": chunk_overlap,
                },
            }
        )

    return chunks


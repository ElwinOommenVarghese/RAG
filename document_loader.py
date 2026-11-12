"""
Stage 1 of Ingestion: Document Loading & Text Extraction.
Supports PDF (via PyMuPDF / fitz) and plain-text (.txt) files.
"""

from pathlib import Path
from typing import Tuple
import pymupdf


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def extract_text_from_file(file_path: Path) -> Tuple[str, int, str]:
    """
    Extract raw text from a PDF or TXT file.

    Returns:
        (raw_text, page_count, file_type)

    Raises:
        ValueError: If file format is unsupported, file is empty, or extraction fails.
    """
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path.name}")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Please upload a .pdf or .txt file."
        )

    if file_path.stat().st_size == 0:
        raise ValueError(f"The uploaded file '{file_path.name}' is empty (0 bytes).")

    if ext == ".pdf":
        return _extract_pdf(file_path)
    return _extract_txt(file_path)


def _extract_pdf(file_path: Path) -> Tuple[str, int, str]:
    """Extract text page by page from a PDF using PyMuPDF."""
    try:
        doc = pymupdf.open(str(file_path))
    except Exception as exc:
        raise ValueError(f"Failed to open PDF '{file_path.name}': {exc}") from exc

    try:
        page_count = len(doc)
        if page_count == 0:
            raise ValueError(f"PDF '{file_path.name}' has 0 pages.")

        pages_text = []
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text") or ""
            if text.strip():
                pages_text.append(f"[Page {page_num + 1}]\n{text}")

        combined = "\n\n".join(pages_text).strip()
        if not combined:
            raise ValueError(
                f"PDF '{file_path.name}' appears to be empty or scanned (no selectable text found)."
            )
        return combined, page_count, "pdf"
    finally:
        doc.close()


def _extract_txt(file_path: Path) -> Tuple[str, int, str]:
    """Extract text from a TXT file with UTF-8 fallback to Latin-1."""
    raw_bytes = file_path.read_bytes()
    text = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None or not text.strip():
        raise ValueError(f"Text file '{file_path.name}' is empty or unreadable.")

    # Estimate logical pages (~3000 characters per page) for display consistency
    estimated_pages = max(1, (len(text) + 2999) // 3000)
    return text.strip(), estimated_pages, "txt"


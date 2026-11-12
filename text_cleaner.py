"""
Stage 2 of Ingestion: Text Cleaning & Normalization.
Removes noisy whitespace, null bytes, and excessive blank lines while
preserving paragraph boundaries for clean chunking.
"""

import re


def clean_text(raw_text: str) -> str:
    """Normalize extracted document text while keeping paragraph structure."""
    if not raw_text:
        return ""

    # Remove null bytes and normalize line endings
    text = raw_text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple spaces/tabs on the same line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse 3+ consecutive newlines down to 2 (a single paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


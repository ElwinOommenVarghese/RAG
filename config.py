"""
Central configuration for Campus Companion – Interactive RAG Learning Lab.
Loads environment variables from `.env` and defines default RAG parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DOCS_DIR = BASE_DIR / "sample_documents"
REGISTRY_PATH = DATA_DIR / "documents_registry.json"
ACTIVITY_LOG_PATH = DATA_DIR / "activity_log.json"

# Ensure required runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load credentials from .env (never hardcode API keys)
load_dotenv(dotenv_path=BASE_DIR / ".env")


def get_groq_api_key() -> str:
    """Return the configured Groq API key, or empty string if missing/placeholder."""
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        return ""
    return key


def get_groq_model() -> str:
    """Return the configured Groq model name."""
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()


# RAG Default Settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "campus_companion_docs"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.0  # 0.0 means return all top-K without filtering


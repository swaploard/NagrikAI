from __future__ import annotations

from pathlib import Path

CHROMA_PERSIST_DIR = Path("chroma_db")
CONTENT_DIR = Path("content")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:latest"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5

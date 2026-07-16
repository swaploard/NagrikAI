from __future__ import annotations

from pathlib import Path

from nagrik_ai.utils.markdown_utils import list_markdown_files, load_markdown
from nagrik_ai.utils.text_utils import chunk_markdown
from nagrik_ai.vectorstore.chroma_store import ChromaStore


class Vectorizer:
    def __init__(self, chroma_store: ChromaStore, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chroma_store = chroma_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def vectorize_directory(self, directory: Path, site: str = "") -> int:
        files = list_markdown_files(directory)
        total_chunks = 0
        for filepath in files:
            text = load_markdown(filepath)
            chunks = chunk_markdown(text, self.chunk_size, self.chunk_overlap)
            metadatas = [{"source": str(filepath), "site": site, "chunk": i} for i in range(len(chunks))]
            self.chroma_store.add_texts(chunks, metadatas)
            total_chunks += len(chunks)
        return total_chunks

    def vectorize_texts(self, texts: list[str], metadatas: list[dict[str, object]] | None = None) -> list[str]:
        return self.chroma_store.add_texts(texts, metadatas)

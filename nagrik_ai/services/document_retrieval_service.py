from __future__ import annotations

from typing import Any

from nagrik_ai.vectorstore.chroma_store import ChromaStore


class DocumentRetrievalService:
    def __init__(self, chroma_store: ChromaStore, top_k: int = 5) -> None:
        self.chroma_store = chroma_store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        docs = self.chroma_store.similarity_search(query, k=self.top_k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]

    def format_context(self, results: list[dict[str, Any]]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            metadata = r["metadata"]
            source = metadata.get("source", "unknown")
            site = metadata.get("site", "unknown")
            parts.append(f"[{i}] Source: {source} (Site: {site})\n{r['content']}")
        return "\n\n".join(parts)

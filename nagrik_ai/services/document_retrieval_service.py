from __future__ import annotations

from typing import Any, cast

from nagrik_ai.vectorstore.chroma_store import ChromaStore


class DocumentRetrievalService:
    def __init__(self, chroma_store: ChromaStore, top_k: int = 5) -> None:
        self.chroma_store = chroma_store
        self.top_k = top_k

    def _normalize_metadata(self, metadata: Any) -> dict[str, Any]:
        if isinstance(metadata, dict):
            normalized: dict[str, Any] = {}
            mapping = cast(dict[Any, Any], metadata)
            for key, value in mapping.items():
                if isinstance(key, str):
                    normalized[key] = value
            return normalized
        return {}

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        docs = self.chroma_store.query(query, n_results=self.top_k)
        result: list[dict[str, Any]] = []
        for doc in docs:
            metadata = self._normalize_metadata(getattr(doc, "metadata", None))
            content = getattr(doc, "page_content", "")
            result.append({"content": content, "metadata": metadata})
        return result

    def format_context(self, results: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            metadata = self._normalize_metadata(r.get("metadata", {}))
            source = str(metadata.get("source", "unknown"))
            site = str(metadata.get("site", "unknown"))
            content = str(r.get("content", ""))
            parts.append(f"[{i}] Source: {source} (Site: {site})\n{content}")
        return "\n\n".join(parts)

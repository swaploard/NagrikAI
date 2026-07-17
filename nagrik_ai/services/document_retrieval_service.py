from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.documents import Document

from nagrik_ai.services.reranker import Reranker
from nagrik_ai.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class DocumentRetrievalService:
    def __init__(
        self,
        chroma_store: ChromaStore,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
        reranker: Reranker | None = None,
    ) -> None:
        self.chroma_store = chroma_store
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.reranker = reranker
        logger.info(
            "Initialized document retrieval service with top_k=%d, fetch_k=%d, lambda_mult=%.2f, reranker=%s",
            top_k,
            fetch_k,
            lambda_mult,
            reranker.model_name if reranker else "disabled",
        )

    def _normalize_metadata(self, metadata: Any) -> dict[str, Any]:
        if isinstance(metadata, dict):
            normalized: dict[str, Any] = {}
            mapping = cast(dict[Any, Any], metadata)
            for key, value in mapping.items():
                if isinstance(key, str):
                    normalized[key] = value
            return normalized
        return {}

    def _docs_to_dicts(self, docs: list[Document]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for doc in docs:
            metadata = self._normalize_metadata(getattr(doc, "metadata", None))
            content = getattr(doc, "page_content", "")
            result.append({"content": content, "metadata": metadata})
        return result

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        if self.reranker is not None:
            docs = self.chroma_store.similarity_search(query, k=self.fetch_k)
            results = self._docs_to_dicts(docs)
            return self.reranker.rerank(query, results, top_k=self.top_k)

        docs = self.chroma_store.query(
            query,
            n_results=self.top_k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )
        return self._docs_to_dicts(docs)

    def format_context(self, results: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            metadata = self._normalize_metadata(r.get("metadata", {}))
            source = str(metadata.get("source", "unknown"))
            site = str(metadata.get("site", "unknown"))
            content = str(r.get("content", ""))
            parts.append(f"[{i}] Source: {source} (Site: {site})\n{content}")
        return "\n\n".join(parts)

from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.documents import Document

from nagrik_ai.services.reranker import Reranker
from nagrik_ai.vectorstore.bm25_retriever import BM25Retriever
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
        hybrid_search: bool = False,
        bm25_retriever: BM25Retriever | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.chroma_store = chroma_store
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.reranker = reranker
        self.hybrid_search = hybrid_search
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        logger.info(
            "Initialized document retrieval service with top_k=%d, fetch_k=%d, "
            "lambda_mult=%.2f, reranker=%s, hybrid=%s",
            top_k,
            fetch_k,
            lambda_mult,
            reranker.model_name if reranker else "disabled",
            hybrid_search,
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

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        for rank, doc in enumerate(dense_results, 1):
            key = doc["content"]
            scores[key] = 1.0 / (self.rrf_k + rank)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_results, 1):
            key = doc["content"]
            if key in scores:
                scores[key] += 1.0 / (self.rrf_k + rank)
            else:
                scores[key] = 1.0 / (self.rrf_k + rank)
                doc_map[key] = doc

        sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
        return [doc_map[k] for k in sorted_keys]

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        if self.hybrid_search and self.bm25_retriever is not None:
            dense_docs = self.chroma_store.similarity_search(query, k=self.fetch_k)
            dense_results = self._docs_to_dicts(dense_docs)
            bm25_results = self.bm25_retriever.retrieve(query, k=self.fetch_k)
            fused = self._reciprocal_rank_fusion(dense_results, bm25_results)

            if self.reranker is not None:
                return self.reranker.rerank(query, fused, top_k=self.top_k)
            return fused[: self.top_k]

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

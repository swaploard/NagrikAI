from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from nagrik_ai.vectorstore.chroma_store import ChromaStore, validate_metadata

logger = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(
        self,
        chroma_store: ChromaStore,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chroma_store = chroma_store
        self.k1 = k1
        self.b = b
        self._bm25: Any = None
        self._documents: list[dict[str, Any]] = []

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _build_index(self) -> None:
        build_start = time.perf_counter()
        docs: list[Document] = self.chroma_store.get_all_documents()
        self._documents = []
        for doc in docs:
            metadata: dict[str, Any] = {}
            raw_metadata: dict[str, Any] | None = getattr(doc, "metadata", None)
            if isinstance(raw_metadata, dict):
                metadata = validate_metadata(raw_metadata)
            self._documents.append(
                {
                    "content": doc.page_content,
                    "metadata": metadata,
                }
            )
        if not docs:
            self._bm25 = None
            logger.warning("No documents found in ChromaDB, BM25 index not built")
            return
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in docs]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        build_elapsed = (time.perf_counter() - build_start) * 1000
        logger.info(
            "Built BM25 index with %d documents (k1=%.2f, b=%.2f) in %.0f ms",
            len(self._documents),
            self.k1,
            self.b,
            build_elapsed,
        )

    def retrieve(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        if self._bm25 is None:
            self._build_index()

        if self._bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        bm25: Any = self._bm25
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]
        results = [dict(self._documents[i]) for i in top_indices]
        for i, index in enumerate(top_indices):
            results[i]["score"] = float(scores[index])
        return results

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Protocol

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from nagrik_ai.vectorstore.chroma_store import validate_metadata

logger = logging.getLogger(__name__)


class DocumentSource(Protocol):
    def get_all_documents(self) -> list[Document]: ...


class BM25Retriever:
    def __init__(
        self,
        chroma_store: DocumentSource,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chroma_store = chroma_store
        self.k1 = k1
        self.b = b
        self._bm25: Any = None
        self._documents: list[dict[str, Any]] = []

    def invalidate(self) -> None:
        self._bm25 = None
        self._documents = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+(?:[/.-][a-z0-9]+)*", text.casefold())

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

    def retrieve(self, query: str, k: int = 20, scope: str = "all_versions") -> list[dict[str, Any]]:
        if self._bm25 is None:
            self._build_index()

        if self._bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        bm25: Any = self._bm25
        scores = bm25.get_scores(tokenized_query)
        eligible = [
            index
            for index, document in enumerate(self._documents)
            if self._in_scope(document["metadata"], scope)
        ]
        exact_tokens = set(tokenized_query)

        def identifier_bonus(index: int) -> float:
            metadata = self._documents[index]["metadata"]
            identifiers = " ".join(
                str(metadata.get(field, ""))
                for field in (
                    "document_number",
                    "hsn_sac_codes",
                    "amendment_document_ids",
                    "referenced_document_urls",
                    "logical_location",
                )
            )
            return 2.0 * len(exact_tokens & set(self._tokenize(identifiers)))

        top_indices = sorted(
            eligible,
            key=lambda i: float(scores[i]) + identifier_bonus(i),
            reverse=True,
        )[:k]
        results = [dict(self._documents[i]) for i in top_indices]
        for i, index in enumerate(top_indices):
            results[i]["score"] = float(scores[index])
            results[i]["metadata"]["retrieval_scope"] = scope
        return results

    @staticmethod
    def _in_scope(metadata: dict[str, Any], scope: str) -> bool:
        if scope == "all_versions":
            return True
        raw = metadata.get("retrieval_scopes", "[]")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                scopes = parsed if isinstance(parsed, list) else [raw]
            except json.JSONDecodeError:
                scopes = [raw]
        else:
            scopes = raw if isinstance(raw, list) else []
        if scope == "legally_applicable":
            return metadata.get("legal_applicability_status") == "applicable" and scope in scopes
        return scope in scopes or metadata.get("source_version_status") == scope

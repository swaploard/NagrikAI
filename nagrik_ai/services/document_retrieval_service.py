from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.documents import Document

from nagrik_ai.services.reranker import Reranker
from nagrik_ai.vectorstore.bm25_retriever import BM25Retriever
from nagrik_ai.vectorstore.chroma_store import ChromaStore, validate_metadata

logger = logging.getLogger(__name__)


def get_score(doc: Document | dict[str, Any]) -> float:
    """Safely extract score from various retriever output formats.

    Handles:
    - dict with "score" key
    - objects with "score" attribute (e.g., LangChain Document)
    - missing scores default to 0.0
    """
    score = doc.get("score", 0.0) if isinstance(doc, dict) else getattr(doc, "score", 0.0)

    if isinstance(score, (int, float)):
        return float(score)
    return 0.0


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

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            normalized[key] = value
        return normalized

    def _docs_to_dicts(self, docs: list[Document]) -> list[dict[str, Any]]:
        logger.debug("Converting %d documents to dicts", len(docs))
        result: list[dict[str, Any]] = []
        for doc in docs:
            metadata = validate_metadata(cast("dict[str, Any]", getattr(doc, "metadata", {})))
            content = getattr(doc, "page_content", "")
            result.append({"content": content, "metadata": metadata})
        logger.debug("Converted %d documents, content lengths: %s", len(result), [len(d["content"]) for d in result])
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
        logger.debug(
            "Retrieving documents for query: %r (top_k=%d, hybrid=%s, reranker=%s)",
            query,
            self.top_k,
            self.hybrid_search,
            self.reranker is not None,
        )
        if self.hybrid_search and self.bm25_retriever is not None:
            dense_docs = self.chroma_store.similarity_search(query, k=self.fetch_k)
            dense_results = self._docs_to_dicts(dense_docs)
            bm25_results = self.bm25_retriever.retrieve(query, k=self.fetch_k)
            fused = self._reciprocal_rank_fusion(dense_results, bm25_results)
            logger.debug(
                "Hybrid search: dense=%d, bm25=%d, fused=%d",
                len(dense_results),
                len(bm25_results),
                len(fused),
            )

            if self.reranker is not None:
                reranked = self.reranker.rerank(query, fused, top_k=self.top_k)
                logger.debug("Reranked to %d results", len(reranked))
                return reranked
            fused.sort(key=get_score, reverse=True)
            return fused[: self.top_k]

        if self.reranker is not None:
            docs = self.chroma_store.similarity_search(query, k=self.fetch_k)
            results = self._docs_to_dicts(docs)
            reranked = self.reranker.rerank(query, results, top_k=self.top_k)
            logger.debug("Reranked similarity search to %d results", len(reranked))
            return reranked

        docs = self.chroma_store.query(
            query,
            n_results=self.top_k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )
        results = self._docs_to_dicts(docs)
        results.sort(key=get_score, reverse=True)
        logger.debug("Vector query returned %d results", len(results))
        return results

    def format_context_block(self, doc: dict[str, Any], citation_id: int | None = None) -> str:
        """Format a single document as a structured context block.

        Args:
            doc: Document dictionary with content and metadata
            citation_id: Optional explicit citation ID (for backward compatibility).
                         If not provided, uses doc["citation_id"] if set.
        """
        logger.debug("Formatting context block with citation_id=%s", citation_id)
        metadata = self._normalize_metadata(doc.get("metadata", {}))
        source_id = str(metadata.get("source_id", metadata.get("source", "unknown")))
        title = str(metadata.get("title", "Unknown"))
        url = str(metadata.get("citation_url", metadata.get("url", "unknown")))
        domain = str(metadata.get("domain", "unknown"))
        content = str(doc.get("content", ""))

        # Use locked citation_id from doc, fallback to passed citation_id
        index = doc.get("citation_id", citation_id or 1)

        logger.debug(
            "Block %d metadata: source_id=%s, title=%s, url=%s, domain=%s, content_len=%d",
            index, source_id, title, url, domain, len(content),
        )

        header = (
            f"[{index}] Source ID: {source_id} | Title: {title} | "
            f"URL: {url} | Domain: {domain}"
        )
        return f"{header}\nContent: {content}"

    def format_context(self, results: list[dict[str, Any]]) -> tuple[str, dict[int, dict[str, Any]]]:
        """Format results as context string with locked citation IDs.

        Args:
            results: List of document dicts from retrieve()

        Returns:
            Tuple of (formatted_context_string, citation_mapping)
            - formatted_context_string: Ready for LLM prompt
            - citation_mapping: dict mapping citation_id -> original doc dict
              for source resolution in UI/citations
        """
        logger.debug("Formatting context from %d results", len(results))

        # LOCK citation IDs BEFORE any sorting/filtering
        citation_mapping: dict[int, dict[str, Any]] = {}
        for i, doc in enumerate(results, start=1):
            doc["citation_id"] = i
            citation_mapping[i] = doc

        # Sort by score descending for display (doesn't affect citation IDs)
        sorted_results = sorted(results, key=get_score, reverse=True)
        blocks = [self.format_context_block(doc) for doc in sorted_results]
        context = "\n\n---\n\n".join(blocks)
        logger.debug("Formatted context length: %d chars, %d blocks", len(context), len(blocks))
        return context, citation_mapping

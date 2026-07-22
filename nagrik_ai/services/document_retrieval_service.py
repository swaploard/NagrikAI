"""Document retrieval service coordinating vector, BM25, and hybrid search."""

from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.documents import Document

from nagrik_ai.services.citation_service import format_context_block as fmt_block
from nagrik_ai.services.reranker import Reranker
from nagrik_ai.services.tracing import LangSmithTracer, get_tracer
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


def _flatten_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Flatten metadata into top-level keys for citation_service.format_context_block."""
    metadata = doc.get("metadata", {})
    flat = {
        "content": doc.get("content", ""),
        "page_content": doc.get("content", ""),
        "source_id": metadata.get("source_id", metadata.get("source", "unknown")),
        "title": metadata.get("title", "Unknown"),
        "url": metadata.get("citation_url", metadata.get("url", "unknown")),
        "domain": metadata.get("domain", "unknown"),
        "chunk_index": metadata.get("chunk_index", 0),
        "total_chunks": metadata.get("total_chunks", 1),
        "score": doc.get("score", 0.0),
        "citation_id": doc.get("citation_id", 1),
    }
    return flat


def _log_retrieved_docs(results: list[dict[str, Any]]) -> None:
    for i, doc in enumerate(results, start=1):
        flat = _flatten_doc(doc)
        content = flat.get("content", "") or flat.get("page_content", "")
        preview = content[:200].replace("\n", " ")
        logger.info(
            "Doc %d: source_id=%s title=%s chunk=%d/%d url=%s preview=%s",
            i,
            flat["source_id"],
            flat["title"],
            flat["chunk_index"],
            flat["total_chunks"],
            flat["url"],
            preview,
        )


def _extract_doc_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    """Extract a clean metadata dict for tracing from a result doc."""
    flat = _flatten_doc(doc)
    return {
        "source_id": flat["source_id"],
        "chunk_index": flat["chunk_index"],
        "total_chunks": flat["total_chunks"],
        "title": flat["title"],
        "url": flat["url"],
        "domain": flat["domain"],
        "score": flat["score"],
    }


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
        tracer: LangSmithTracer | None = None,
    ) -> None:
        self.chroma_store = chroma_store
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.reranker = reranker
        self.hybrid_search = hybrid_search
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.tracer: LangSmithTracer = tracer or get_tracer()
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

    def _get_doc_metadata(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        metadata_list: list[dict[str, Any]] = []
        for doc in results:
            flat = _flatten_doc(doc)
            metadata_list.append(
                {
                    "citation_id": flat["citation_id"],
                    "source_id": flat["source_id"],
                    "title": flat["title"],
                    "url": flat["url"],
                    "domain": flat["domain"],
                    "score": flat["score"],
                    "chunk_index": flat["chunk_index"],
                }
            )
        return metadata_list

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        if self.hybrid_search and self.bm25_retriever is not None:
            strategy = "hybrid+reranker" if self.reranker is not None else "hybrid"
        elif self.reranker is not None:
            strategy = "dense+reranker"
        else:
            strategy = "dense"

        query_metadata: dict[str, Any] = {
            "top_k": self.top_k,
            "fetch_k": self.fetch_k,
            "hybrid": self.hybrid_search,
            "reranker": self.reranker is not None,
            "lambda_mult": self.lambda_mult,
            "strategy": strategy,
        }
        with self.tracer.trace(
            "retrieve_documents",
            "retriever",
            inputs={"query": query},
            metadata=query_metadata,
        ) as span:
            span.start_timer()
            logger.info(
                "Retrieving documents for query (%d chars)",
                len(query),
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
                    logger.info("Retrieved %d documents", len(reranked))
                    _log_retrieved_docs(reranked)
                    latency = span.elapsed_ms()
                    span.set_outputs(
                        {
                            "num_results": len(reranked),
                            "strategy": "hybrid+reranker",
                            "latency_ms": latency,
                            "documents": [_extract_doc_metadata(d) for d in reranked],
                        }
                    )
                    return reranked
                fused.sort(key=get_score, reverse=True)
                result = fused[: self.top_k]
                logger.info("Retrieved %d documents", len(result))
                _log_retrieved_docs(result)
                latency = span.elapsed_ms()
                span.set_outputs(
                    {
                        "num_results": len(result),
                        "strategy": "hybrid",
                        "latency_ms": latency,
                        "documents": [_extract_doc_metadata(d) for d in result],
                    }
                )
                return result

            if self.reranker is not None:
                docs = self.chroma_store.similarity_search(query, k=self.fetch_k)
                results = self._docs_to_dicts(docs)
                reranked = self.reranker.rerank(query, results, top_k=self.top_k)
                logger.info("Retrieved %d documents", len(reranked))
                _log_retrieved_docs(reranked)
                latency = span.elapsed_ms()
                span.set_outputs(
                    {
                        "num_results": len(reranked),
                        "strategy": "dense+reranker",
                        "latency_ms": latency,
                        "documents": [_extract_doc_metadata(d) for d in reranked],
                    }
                )
                return reranked

            docs = self.chroma_store.query(
                query,
                n_results=self.top_k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult,
            )
            results = self._docs_to_dicts(docs)
            results.sort(key=get_score, reverse=True)
            logger.info("Retrieved %d documents", len(results))
            _log_retrieved_docs(results)
            latency = span.elapsed_ms()
            span.set_outputs(
                {
                    "num_results": len(results),
                    "strategy": "dense",
                    "latency_ms": latency,
                    "documents": [_extract_doc_metadata(d) for d in results],
                }
            )
            return results

    def format_context_block(self, doc: dict[str, Any], citation_id: int | None = None) -> str:
        """Format a single document as a structured context block (delegates to citation_service)."""
        # Flatten metadata into top-level keys expected by citation_service
        flat_doc = _flatten_doc(doc)
        index = flat_doc.get("citation_id", citation_id or 1)
        return fmt_block(flat_doc, index)

    def format_context(self, results: list[dict[str, Any]]) -> tuple[str, dict[int, dict[str, Any]]]:
        """Format results as context string with locked citation IDs.

        If citation_ids are not already locked (by orchestrator), lock them here
        for backward compatibility.
        """
        logger.debug("Formatting context from %d results", len(results))

        # LOCK citation IDs if not already locked (backward compatibility)
        for i, doc in enumerate(results, start=1):
            if "citation_id" not in doc:
                doc["citation_id"] = i

        # Sort by score descending for display (doesn't affect citation IDs)
        sorted_results = sorted(results, key=get_score, reverse=True)
        blocks = [self.format_context_block(doc) for doc in sorted_results]
        context = "\n\n---\n\n".join(blocks)
        logger.debug("Formatted context length: %d chars, %d blocks", len(context), len(blocks))

        # Build citation mapping from locked IDs
        citation_mapping: dict[int, dict[str, Any]] = {}
        for doc in results:
            cid = doc.get("citation_id")
            if cid is not None:
                citation_mapping[cid] = doc

        return context, citation_mapping

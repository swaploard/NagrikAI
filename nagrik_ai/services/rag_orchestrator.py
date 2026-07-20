"""RAG orchestrator coordinating retrieval, citation management, and LLM generation."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from typing import Any

from nagrik_ai.models.rag_result import RAGResult, SourceInfo
from nagrik_ai.prompts.prompt_loader import load_prompt
from nagrik_ai.services.citation_service import (
    assign_citation_ids,
    deduplicate_for_display,
    extract_snippet,
    flatten_doc,
    format_context_block,
    validate_citations,
)
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.llm_service import BaseLLMService

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Orchestrates the RAG pipeline: retrieve -> format -> generate -> validate."""

    def __init__(
        self,
        retrieval_service: DocumentRetrievalService,
        llm_service: BaseLLMService,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service
        logger.info("Initialized RAG orchestrator")

    def query(self, user_query: str) -> RAGResult:
        """Execute full RAG pipeline and return structured result with citations."""
        start_time = time.perf_counter()

        # Retrieve documents
        results = self.retrieval_service.retrieve(user_query)

        # Build SourceInfo objects with temporary citation_ids for context building
        def get_score(doc: dict[str, Any]) -> float:
            score = doc.get("score", 0.0)
            return float(score) if isinstance(score, (int, float)) else 0.0

        sorted_results = sorted(results, key=get_score, reverse=True)

        # Build SourceInfo objects for all results (before dedup)
        all_sources: list[SourceInfo] = []
        for i, doc in enumerate(sorted_results, start=1):
            flat = flatten_doc(doc)
            all_sources.append(
                SourceInfo(
                    title=str(flat.get("title", "Unknown")),
                    url=str(flat.get("url", "")),
                    domain=str(flat.get("domain", "")),
                    source_id=str(flat.get("source_id", "")),
                    citation_id=i,
                    chunk_index=int(flat.get("chunk_index", 0)),
                    total_chunks=int(flat.get("total_chunks", 1)),
                    score=float(flat.get("score", 0.0)),
                    snippet=extract_snippet(str(flat.get("page_content", flat.get("content", ""))), user_query),
                )
            )

        # Deduplicate BEFORE final citation IDs
        display_sources = deduplicate_for_display(all_sources)

        # Re-assign citation IDs sequentially after dedup
        for i, s in enumerate(display_sources, start=1):
            s.citation_id = i

        # Build context with final citation IDs
        context_blocks: list[str] = []
        for i, s in enumerate(display_sources, start=1):
            # Find the original doc for this source
            orig_doc = next(
                doc
                for doc in sorted_results
                if flatten_doc(doc)["source_id"] == s.source_id and flatten_doc(doc)["chunk_index"] == s.chunk_index
            )
            context_blocks.append(format_context_block(flatten_doc(orig_doc), i))
        context = "\n\n---\n\n".join(context_blocks)

        # Lock citation IDs for citation_map (original mapping)
        _, citation_map = assign_citation_ids(results)

        # Generate response
        system_prompt = load_prompt("system_prompt")
        user_prompt = load_prompt("user_query", question=user_query, context=context)
        response = self.llm_service.generate(user_prompt, system=system_prompt)

        # Validate citations
        citations_valid = validate_citations(response, display_sources)
        logger.debug("LLM response (first 500 chars): %s", response[:500])
        logger.debug("Cited IDs in response: %s", re.findall(r"\[(\d+)\]", response))
        logger.debug("Valid source citation_ids: %s", [s.citation_id for s in display_sources])
        if not citations_valid:
            logger.warning("Citations missing or out of range — appending source list")
            response += "\n\n**Sources:**\n"
            for s in display_sources:
                response += f"[{s.citation_id}] {s.title} - {s.url}\n"

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Observability logging
        logger.info(
            {
                "query": user_query,
                "num_sources": len(display_sources),
                "latency_ms": latency_ms,
                "citation_ids": [s.citation_id for s in display_sources],
                "citations_valid": citations_valid,
            }
        )

        return RAGResult(
            response=response,
            sources=display_sources,
            citation_map=citation_map,
            query=user_query,
            raw_chunks=[str(doc.get("page_content", doc.get("content", ""))) for doc in results],
            latency_ms=latency_ms,
            total_chunks_retrieved=len(results),
            citations_valid=citations_valid,
        )

    def query_stream(self, user_query: str) -> Iterator[dict[str, Any]]:
        """Execute RAG pipeline with streaming LLM response."""
        start_time = time.perf_counter()

        results = self.retrieval_service.retrieve(user_query)

        def get_score(doc: dict[str, Any]) -> float:
            score = doc.get("score", 0.0)
            return float(score) if isinstance(score, (int, float)) else 0.0

        sorted_results = sorted(results, key=get_score, reverse=True)

        # Build SourceInfo objects for all results (before dedup)
        all_sources: list[SourceInfo] = []
        for i, doc in enumerate(sorted_results, start=1):
            flat = flatten_doc(doc)
            all_sources.append(
                SourceInfo(
                    title=str(flat.get("title", "Unknown")),
                    url=str(flat.get("url", "")),
                    domain=str(flat.get("domain", "")),
                    source_id=str(flat.get("source_id", "")),
                    citation_id=i,
                    chunk_index=int(flat.get("chunk_index", 0)),
                    total_chunks=int(flat.get("total_chunks", 1)),
                    score=float(flat.get("score", 0.0)),
                    snippet=extract_snippet(str(flat.get("page_content", flat.get("content", ""))), user_query),
                )
            )

        # Deduplicate BEFORE final citation IDs
        display_sources = deduplicate_for_display(all_sources)

        # Re-assign citation IDs sequentially after dedup
        for i, s in enumerate(display_sources, start=1):
            s.citation_id = i

        # Build context with final citation IDs
        context_blocks: list[str] = []
        for i, s in enumerate(display_sources, start=1):
            orig_doc = next(
                doc
                for doc in sorted_results
                if flatten_doc(doc)["source_id"] == s.source_id and flatten_doc(doc)["chunk_index"] == s.chunk_index
            )
            context_blocks.append(format_context_block(flatten_doc(orig_doc), i))
        context = "\n\n---\n\n".join(context_blocks)

        # Lock citation IDs for citation_map (original mapping)
        _, citation_map = assign_citation_ids(results)

        system_prompt = load_prompt("system_prompt")
        user_prompt = load_prompt("user_query", question=user_query, context=context)

        # Stream tokens
        full_response = ""
        for chunk in self.llm_service.generate_stream(user_prompt, system=system_prompt):
            full_response += chunk
            yield {"type": "token", "content": chunk}

        # Post-generation citation validation
        citations_valid = validate_citations(full_response, display_sources)
        if not citations_valid:
            logger.warning("Citations missing or out of range — appending source list")
            full_response += "\n\n**Sources:**\n"
            for s in display_sources:
                full_response += f"[{s.citation_id}] {s.title} - {s.url}\n"

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            {
                "query": user_query,
                "num_sources": len(display_sources),
                "latency_ms": latency_ms,
                "citation_ids": [s.citation_id for s in display_sources],
                "citations_valid": citations_valid,
            }
        )

        yield {
            "type": "final",
            "data": RAGResult(
                response=full_response,
                sources=display_sources,
                citation_map=citation_map,
                query=user_query,
                raw_chunks=[str(doc.get("page_content", doc.get("content", ""))) for doc in results],
                latency_ms=latency_ms,
                total_chunks_retrieved=len(results),
                citations_valid=citations_valid,
            ),
        }

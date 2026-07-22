from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from typing import Any

from nagrik_ai.models.rag_result import RAGResult, SourceInfo
from nagrik_ai.prompts.prompt_loader import get_prompt_version, load_prompt
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
from nagrik_ai.services.tracing import LangSmithTracer, get_tracer

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    def __init__(
        self,
        retrieval_service: DocumentRetrievalService,
        llm_service: BaseLLMService,
        tracer: LangSmithTracer | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service
        self.tracer: LangSmithTracer = tracer or get_tracer()
        logger.info("Initialized RAG orchestrator")

    def query(
        self,
        user_query: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> RAGResult:
        logger.info("Query: %s", user_query)
        with self.tracer.trace(
            "rag_query",
            "chain",
            inputs={"query": user_query},
            metadata={
                "pipeline": "rag",
                "session_id": session_id,
                "user_id": user_id,
            },
            session_id=session_id,
            user_id=user_id,
        ) as span:
            span.start_timer()
            start_time = time.perf_counter()

            logger.info("Retrieving relevant documents")
            results = self.retrieval_service.retrieve(user_query)

            def get_score(doc: dict[str, Any]) -> float:
                score = doc.get("score", 0.0)
                return float(score) if isinstance(score, (int, float)) else 0.0

            sorted_results = sorted(results, key=get_score, reverse=True)

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

            display_sources = deduplicate_for_display(all_sources)

            for i, s in enumerate(display_sources, start=1):
                s.citation_id = i

            context_blocks: list[str] = []
            for i, s in enumerate(display_sources, start=1):
                orig_doc = next(
                    doc
                    for doc in sorted_results
                    if flatten_doc(doc)["source_id"] == s.source_id and flatten_doc(doc)["chunk_index"] == s.chunk_index
                )
                context_blocks.append(format_context_block(flatten_doc(orig_doc), i))
            context = "\n\n---\n\n".join(context_blocks)

            _, citation_map = assign_citation_ids(results)

            # --- Prompt assembly and tracing ---
            system_prompt = load_prompt("system_prompt")
            user_prompt = load_prompt("user_query", question=user_query, context=context)
            prompt_versions = {
                "system_prompt": get_prompt_version("system_prompt"),
                "user_query": get_prompt_version("user_query"),
            }
            with self.tracer.trace(
                "format_prompt",
                "prompt",
                inputs={
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "question": user_query,
                },
                metadata={"prompt_versions": prompt_versions},
                session_id=session_id,
                user_id=user_id,
            ):
                pass

            # --- LLM generation ---
            logger.info("Generating response with LLM")
            response = self.llm_service.generate(user_prompt, system=system_prompt)

            citations_valid = validate_citations(response, display_sources)
            logger.info("Response: %s", response[:200])
            logger.debug("Cited IDs in response: %s", re.findall(r"\[(\d+)\]", response))
            logger.debug("Valid source citation_ids: %s", [s.citation_id for s in display_sources])
            if not citations_valid:
                logger.warning("Citations missing or out of range — appending source list")
                response += "\n\n**Sources:**\n"
                for s in display_sources:
                    response += f"[{s.citation_id}] {s.title} - {s.url}\n"

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

            result = RAGResult(
                response=response,
                sources=display_sources,
                citation_map=citation_map,
                query=user_query,
                raw_chunks=[str(doc.get("page_content", doc.get("content", ""))) for doc in results],
                latency_ms=latency_ms,
                total_chunks_retrieved=len(results),
                citations_valid=citations_valid,
            )

            cited_ids = [int(m) for m in re.findall(r"\[(\d+)\]", response)]
            span.set_outputs(
                {
                    "response": response,
                    "num_sources": len(display_sources),
                    "latency_ms": latency_ms,
                    "citations_valid": citations_valid,
                    "response_length": len(response),
                    "citations": [
                        {
                            "citation_id": s.citation_id,
                            "source_id": s.source_id,
                            "title": s.title,
                            "url": s.url,
                            "domain": s.domain,
                            "score": s.score,
                        }
                        for s in display_sources
                    ],
                    "cited_ids_in_response": cited_ids,
                }
            )
            return result

    def query_stream(
        self,
        user_query: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        logger.info("Query: %s", user_query)
        with self.tracer.trace(
            "rag_query_stream",
            "chain",
            inputs={"query": user_query},
            metadata={
                "pipeline": "rag",
                "streaming": True,
                "session_id": session_id,
                "user_id": user_id,
            },
            session_id=session_id,
            user_id=user_id,
        ) as span:
            span.start_timer()
            start_time = time.perf_counter()

            logger.info("Retrieving relevant documents")
            results = self.retrieval_service.retrieve(user_query)

            def get_score(doc: dict[str, Any]) -> float:
                score = doc.get("score", 0.0)
                return float(score) if isinstance(score, (int, float)) else 0.0

            sorted_results = sorted(results, key=get_score, reverse=True)

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

            display_sources = deduplicate_for_display(all_sources)

            for i, s in enumerate(display_sources, start=1):
                s.citation_id = i

            context_blocks: list[str] = []
            for i, s in enumerate(display_sources, start=1):
                orig_doc = next(
                    doc
                    for doc in sorted_results
                    if flatten_doc(doc)["source_id"] == s.source_id and flatten_doc(doc)["chunk_index"] == s.chunk_index
                )
                context_blocks.append(format_context_block(flatten_doc(orig_doc), i))
            context = "\n\n---\n\n".join(context_blocks)

            _, citation_map = assign_citation_ids(results)

            system_prompt = load_prompt("system_prompt")
            user_prompt = load_prompt("user_query", question=user_query, context=context)
            prompt_versions = {
                "system_prompt": get_prompt_version("system_prompt"),
                "user_query": get_prompt_version("user_query"),
            }
            with self.tracer.trace(
                "format_prompt",
                "prompt",
                inputs={
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "question": user_query,
                },
                metadata={"prompt_versions": prompt_versions},
                session_id=session_id,
                user_id=user_id,
            ):
                pass

            logger.info("Generating streaming response with LLM")
            logger.info("Starting to consume LLM response stream")
            full_response = ""
            for chunk in self.llm_service.generate_stream(user_prompt, system=system_prompt):
                full_response += chunk
                yield {"type": "token", "content": chunk}
            logger.info("Response: %s", full_response[:200])

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

            cited_ids = [int(m) for m in re.findall(r"\[(\d+)\]", full_response)]
            span.set_outputs(
                {
                    "response": full_response,
                    "num_sources": len(display_sources),
                    "latency_ms": latency_ms,
                    "citations_valid": citations_valid,
                    "response_length": len(full_response),
                    "citations": [
                        {
                            "citation_id": s.citation_id,
                            "source_id": s.source_id,
                            "title": s.title,
                            "url": s.url,
                            "domain": s.domain,
                            "score": s.score,
                        }
                        for s in display_sources
                    ],
                    "cited_ids_in_response": cited_ids,
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

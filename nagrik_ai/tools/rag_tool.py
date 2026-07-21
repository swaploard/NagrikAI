"""RAG tool wrapper for the RAG orchestrator."""

import logging
from typing import Any

from nagrik_ai.factories import create_orchestrator
from nagrik_ai.services.llm_service import create_llm_service
from nagrik_ai.services.rag_orchestrator import RAGOrchestrator
from nagrik_ai.services.tracing import get_tracer
from nagrik_ai.tools.web_search import web_search

logger = logging.getLogger(__name__)

_orchestrator: RAGOrchestrator | None = None


def get_orchestrator() -> RAGOrchestrator:
    """Get or create the singleton RAG orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        logger.info("Creating RAG orchestrator singleton")
        _orchestrator = create_orchestrator()
    return _orchestrator


def rag_search(query: str, session_id: str | None = None, user_id: str | None = None) -> str:
    """
    Perform a RAG search and return the generated response.

    Args:
        query: The user's query string.
        session_id: Optional session ID for tracing.
        user_id: Optional user ID for tracing.

    Returns:
        The generated response string with citations.
    """
    orchestrator = get_orchestrator()
    tracer = get_tracer()

    logger.info("RAG search query: %s", query)

    with tracer.trace(
        "rag_search",
        "tool",
        inputs={"query": query},
        metadata={"tool": "rag_search", "session_id": session_id, "user_id": user_id},
        session_id=session_id,
        user_id=user_id,
    ) as span:
        result = orchestrator.query(query, session_id=session_id, user_id=user_id)

        # Fall back to web search if RAG couldn't find an answer
        if "I could not find this information" in result.response:
            logger.info("RAG returned no answer; falling back to web search")
            try:
                web_result = web_search(query)
                llm = create_llm_service()
                synthesis_prompt = (
                    f"Web search results:\n{web_result}\n\n"
                    f"Query: {query}\n\n"
                    f"Provide a comprehensive answer based on these search results."
                )
                synthesized = llm.generate(
                    synthesis_prompt,
                    system="You are a helpful assistant. Answer based on the web search results provided.",
                )
                logger.info("Web search fallback synthesized response successfully")
                span.set_outputs({
                    "response": synthesized,
                    "num_sources": 0,
                    "latency_ms": result.latency_ms,
                    "citations_valid": False,
                    "fallback": "web_search",
                })
                return synthesized
            except Exception:
                logger.exception("Web search fallback failed")
                span.set_outputs({
                    "response": result.response,
                    "num_sources": len(result.sources),
                    "latency_ms": result.latency_ms,
                    "citations_valid": result.citations_valid,
                    "fallback": "failed",
                })
                return result.response

        span.set_outputs({
            "response": result.response,
            "num_sources": len(result.sources),
            "latency_ms": result.latency_ms,
            "citations_valid": result.citations_valid,
        })

        return result.response


def rag_search_with_sources(
    query: str, session_id: str | None = None, user_id: str | None = None
) -> dict[str, Any]:
    """
    Perform a RAG search and return the full result with sources.

    Args:
        query: The user's query string.
        session_id: Optional session ID for tracing.
        user_id: Optional user ID for tracing.

    Returns:
        Dictionary containing response, sources, and metadata.
    """
    orchestrator = get_orchestrator()
    tracer = get_tracer()

    logger.info("RAG search with sources query: %s", query)

    with tracer.trace(
        "rag_search_with_sources",
        "tool",
        inputs={"query": query},
        metadata={"tool": "rag_search_with_sources", "session_id": session_id, "user_id": user_id},
        session_id=session_id,
        user_id=user_id,
    ) as span:
        result = orchestrator.query(query, session_id=session_id, user_id=user_id)

        output: dict[str, Any] = {
            "response": result.response,
            "sources": [
                {
                    "citation_id": s.citation_id,
                    "title": s.title,
                    "url": s.url,
                    "domain": s.domain,
                    "score": s.score,
                }
                for s in result.sources
            ],
            "latency_ms": result.latency_ms,
            "total_chunks_retrieved": result.total_chunks_retrieved,
            "citations_valid": result.citations_valid,
        }

        span.set_outputs(output)
        return output

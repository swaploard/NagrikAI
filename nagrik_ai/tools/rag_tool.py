"""RAG tool wrapper for the LangGraph RAG pipeline."""

import logging
from typing import Any

from nagrik_ai.services.llm_service import create_llm_service
from nagrik_ai.services.tracing import get_tracer
from nagrik_ai.tools.web_search import web_search

logger = logging.getLogger(__name__)


def rag_search(query: str, session_id: str | None = None, user_id: str | None = None) -> str:
    """Perform a RAG search using the LangGraph pipeline and return the generated response."""
    logger.info("RAG search query: %s", query)

    from nagrik_ai.agent.rag_graph import run_rag_query

    result = run_rag_query(query, session_id=session_id, user_id=user_id)
    answer = result.response

    needs_fallback = False
    if not result.citations_valid:
        logger.info("RAG returned invalid citations; falling back to web search")
        needs_fallback = True
    elif "I could not find this information" in answer:
        needs_fallback = True

    if needs_fallback:
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
            return synthesized
        except Exception:
            logger.exception("Web search fallback failed")
            return answer

    return answer


def rag_search_with_sources(query: str, session_id: str | None = None, user_id: str | None = None) -> dict[str, Any]:
    """
    Perform a RAG search and return the full result with sources.

    Args:
        query: The user's query string.
        session_id: Optional session ID for tracing.
        user_id: Optional user ID for tracing.

    Returns:
        Dictionary containing response, sources, and metadata.
    """
    from nagrik_ai.agent.rag_graph import run_rag_query

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
        result = run_rag_query(query, session_id=session_id, user_id=user_id, tracer=tracer)

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

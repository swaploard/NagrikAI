"""Authoritative retrieval tool; fallback decisions belong to the orchestrator."""

from __future__ import annotations

from dataclasses import asdict

from nagrik_ai.models.tool_result import ToolResult
from nagrik_ai.services.tracing import get_tracer
from nagrik_ai.tools.result_utils import failure, tool_result


@tool_result
def rag_search(query: str, session_id: str | None = None, user_id: str | None = None) -> ToolResult:
    from nagrik_ai.agent.rag_graph import run_rag_query

    if not isinstance(query, str) or not query.strip():
        return failure("Search query must not be empty.")
    tracer = get_tracer()
    with tracer.trace("rag_search", "tool", inputs={"query": query}, session_id=session_id, user_id=user_id) as span:
        result = run_rag_query(query, session_id=session_id, user_id=user_id, tracer=tracer, enable_fallback=False)
        sources = [asdict(s) for s in result.sources if s.source_id]
        if not result.citations_valid or not sources or not result.response.strip():
            return failure("No validated retrieval evidence found.", "NOT_FOUND")
        output = {
            "response": result.response,
            "sources": sources,
            "latency_ms": result.latency_ms,
            "total_chunks_retrieved": result.total_chunks_retrieved,
            "citations_valid": result.citations_valid,
        }
        span.set_outputs(output)
        return ToolResult(
            True,
            output,
            None,
            None,
            "authoritative",
            result.latency_ms,
            tuple(dict.fromkeys(s["source_id"] for s in sources)),
        )


# Retain the public import used by existing callers.
rag_search_with_sources = rag_search

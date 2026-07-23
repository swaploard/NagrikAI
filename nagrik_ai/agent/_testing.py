from __future__ import annotations

from nagrik_ai.models.agent_state import AgentState
from nagrik_ai.models.rag_result import RAGResult, SourceInfo


def adapt_to_rag_result(
    state: AgentState,
    query: str = "",
    latency_ms: float = 0.0,
) -> RAGResult:
    sources: list[SourceInfo] = []
    citation_map: dict[int, SourceInfo] = {}

    for i, c in enumerate(state.get("citations", []), start=1):
        src = SourceInfo(
            title=str(c.get("title", "Unknown")),
            url=str(c.get("url", "")),
            domain=str(c.get("domain", "")),
            source_id=str(c.get("source_id", "")),
            citation_id=int(c.get("citation_id", i)),
            chunk_index=int(c.get("chunk_index", 0)),
            total_chunks=int(c.get("total_chunks", 1)),
            score=float(c.get("score", 0.0)),
            snippet=str(c.get("snippet", "")),
        )
        sources.append(src)
        citation_map[src.citation_id] = src

    return RAGResult(
        response=state.get("answer", "") or "",
        sources=sources,
        citation_map=citation_map,
        query=query,
        raw_chunks=[str(doc.get("page_content", doc.get("content", ""))) for doc in state.get("documents", [])],
        latency_ms=latency_ms,
        total_chunks_retrieved=len(state.get("documents", [])),
        citations_valid=len(state.get("errors", [])) == 0,
    )

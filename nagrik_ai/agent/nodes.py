from __future__ import annotations

import logging
from typing import Any

from nagrik_ai.models.agent_state import AgentState
from nagrik_ai.models.rag_result import SourceInfo
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
from nagrik_ai.services.reranker import Reranker

logger = logging.getLogger(__name__)


def retrieve_node(
    state: AgentState,
    retrieval_service: DocumentRetrievalService,
) -> dict[str, Any]:
    docs = retrieval_service.retrieve(state["query"])
    return {
        "documents": docs,
        "metadata": {**state.get("metadata", {}), "retrieved_count": len(docs)},
    }


def rerank_node(
    state: AgentState,
    reranker: Reranker | None = None,
) -> dict[str, Any]:
    if reranker is None:
        return {}
    docs = state["documents"]
    reranked_docs = reranker.rerank(state["query"], docs)
    return {"documents": reranked_docs}


def build_context_node(state: AgentState) -> dict[str, Any]:
    results = state["documents"]
    query = state["query"]

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
                snippet=extract_snippet(
                    str(flat.get("page_content", flat.get("content", ""))),
                    query,
                ),
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

    locked_results, _citation_map = assign_citation_ids(results)

    return {
        "context": context,
        "citations": [
            {
                "citation_id": src.citation_id,
                "source_id": src.source_id,
                "title": src.title,
                "url": src.url,
                "domain": src.domain,
                "score": src.score,
                "snippet": src.snippet,
                "chunk_index": src.chunk_index,
                "total_chunks": src.total_chunks,
            }
            for src in display_sources
        ],
        "documents": locked_results,
    }


def generate_node(
    state: AgentState,
    llm_service: BaseLLMService | None,
    system_prompt: str = "",
) -> dict[str, Any]:
    if llm_service is None:
        error_msg = "llm_service is required for generate_node"
        return {"answer": error_msg, "errors": [error_msg], "candidate_answers": []}
    if not system_prompt:
        system_prompt = load_prompt("system_prompt")
    user_prompt = load_prompt("user_query", question=state["query"], context=state.get("context", "") or "")
    response = llm_service.generate(user_prompt, system=system_prompt)
    return {
        "answer": response,
        "candidate_answers": [response],
    }


def validate_node(state: AgentState) -> dict[str, Any]:
    response_text = state.get("answer", "") or ""
    sources = state.get("citations", [])

    source_infos = [
        SourceInfo(
            title=str(s.get("title", "Unknown")),
            url=str(s.get("url", "")),
            domain=str(s.get("domain", "")),
            source_id=str(s.get("source_id", "")),
            citation_id=int(s.get("citation_id", 0)),
            chunk_index=int(s.get("chunk_index", 0)),
            total_chunks=int(s.get("total_chunks", 1)),
            score=float(s.get("score", 0.0)),
            snippet=str(s.get("snippet", "")),
        )
        for s in sources
    ]

    errors: list[str] = []
    if not source_infos:
        confidence = 0.0
        errors.append("No citation sources available")
    elif validate_citations(response_text, source_infos):
        confidence = 0.8
    else:
        confidence = 0.4
        errors.append("Response contains citations not found in sources")
        source_block = "\n\n**Sources:**\n"
        for s in source_infos:
            source_block += f"[{s.citation_id}] {s.title} - {s.url}\n"
        response_text += source_block

    return {
        "confidence": confidence,
        "errors": errors,
        "answer": response_text,
    }

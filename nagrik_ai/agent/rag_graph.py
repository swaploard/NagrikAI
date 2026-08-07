from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nagrik_ai.agent.nodes import (
    build_context_node,
    classify_node,
    finalize_node,
    generate_node,
    generate_stream_node,
    rerank_node,
    retrieve_node,
    validate_node,
)
from nagrik_ai.models.agent_state import AgentState
from nagrik_ai.models.rag_result import RAGResult
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.llm_service import BaseLLMService
from nagrik_ai.services.reranker import Reranker
from nagrik_ai.services.tracing import LangSmithTracer, get_tracer
from nagrik_ai.tools.web_search import web_search

logger = logging.getLogger(__name__)


def build_initial_state(
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    retrieval_config: dict[str, Any] | None = None,
) -> AgentState:
    return AgentState(
        query=query,
        rewritten_queries=[],
        documents=[],
        candidate_answers=[],
        answer=None,
        confidence=None,
        citations=[],
        errors=[],
        metadata={},
        tool_calls=[],
        tool_results=[],
        current_tool=None,
        session_id=session_id,
        user_id=user_id,
        trace_id=None,
        context=None,
        retrieval_config=retrieval_config or {},
        citations_valid=None,
        messages=[],
        rag_result=None,
        _streaming_buffer=[],
        _streaming_callback=None,
    )


def create_rag_graph(
    retrieval_service: DocumentRetrievalService,
    reranker: Reranker | None = None,
    llm_service: BaseLLMService | None = None,
    system_prompt: str = "",
    tracer: LangSmithTracer | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    enable_streaming: bool = False,
    enable_fallback: bool = False,
    enable_self_correction: bool = False,
    max_retries: int = 2,
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    _tracer = tracer or get_tracer()

    if llm_service is None:
        from nagrik_ai.services.llm_service import create_llm_service

        llm_service = create_llm_service(temperature=0.1)
    workflow: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(
        state_schema=AgentState,
        context_schema=None,
    )

    def wrapped_classify(state: AgentState) -> dict[str, Any]:
        return classify_node(
            state, tracer=_tracer, session_id=state.get("session_id"), user_id=state.get("user_id")
        )

    def wrapped_retrieve(state: AgentState) -> dict[str, Any]:
        return retrieve_node(
            state, retrieval_service, tracer=_tracer, session_id=state.get("session_id"), user_id=state.get("user_id")
        )

    def wrapped_rerank(state: AgentState) -> dict[str, Any]:
        return rerank_node(
            state, reranker, tracer=_tracer, session_id=state.get("session_id"), user_id=state.get("user_id")
        )

    def wrapped_generate(state: AgentState) -> dict[str, Any]:
        return generate_node(
            state,
            llm_service,
            system_prompt,
            tracer=_tracer,
            session_id=state.get("session_id"),
            user_id=state.get("user_id"),
        )

    def wrapped_generate_stream(state: AgentState) -> dict[str, Any]:
        return generate_stream_node(
            state,
            llm_service,
            system_prompt,
            tracer=_tracer,
            session_id=state.get("session_id"),
            user_id=state.get("user_id"),
        )

    def wrapped_build_context(state: AgentState) -> dict[str, Any]:
        return build_context_node(
            state, tracer=_tracer, session_id=state.get("session_id"), user_id=state.get("user_id")
        )

    def wrapped_validate(state: AgentState) -> dict[str, Any]:
        return validate_node(state, tracer=_tracer, session_id=state.get("session_id"), user_id=state.get("user_id"))

    def wrapped_finalize(state: AgentState) -> dict[str, Any]:
        return finalize_node(state, tracer=_tracer, session_id=state.get("session_id"), user_id=state.get("user_id"))

    def web_search_fallback_node(state: AgentState) -> dict[str, Any]:
        """Web search fallback when RAG fails to find relevant information."""
        query = state.get("query", "")
        _tracer = tracer or get_tracer()
        with _tracer.trace(
            "web_search_fallback",
            "tool",
            inputs={"query": query},
            session_id=state.get("session_id"),
            user_id=state.get("user_id"),
        ):
            web_result = web_search(query)
            synthesis_prompt = (
                f"Web search results:\n{web_result}\n\n"
                f"Query: {query}\n\n"
                f"Provide a comprehensive answer based on these search results."
            )
            synthesized = llm_service.generate(
                synthesis_prompt,
                system="You are a helpful assistant. Answer based on the web search results provided.",
            )
            return {"answer": synthesized, "errors": []}

    workflow.add_node("classify", wrapped_classify)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("retrieve", wrapped_retrieve)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("rerank", wrapped_rerank)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("build_context", wrapped_build_context)  # pyright: ignore[reportUnknownMemberType]
    generate_node_name = "generate_stream" if enable_streaming else "generate"
    workflow.add_node("generate", wrapped_generate)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("generate_stream", wrapped_generate_stream)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("validate", wrapped_validate)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("finalize", wrapped_finalize)  # pyright: ignore[reportUnknownMemberType]

    if enable_fallback:
        workflow.add_node("web_search_fallback", web_search_fallback_node)  # pyright: ignore[reportUnknownMemberType]

    if enable_self_correction:
        _generate_fn = wrapped_generate_stream if enable_streaming else wrapped_generate

        def _retry_generate(state: AgentState) -> dict[str, Any]:
            metadata = dict(state.get("metadata", {}))
            metadata["retry_count"] = metadata.get("retry_count", 0) + 1
            result = _generate_fn(state)
            result["metadata"] = {
                **state.get("metadata", {}),
                **result.get("metadata", {}),
                "retry_count": metadata["retry_count"],
            }
            return result

        workflow.add_node("retry_generate", _retry_generate)  # pyright: ignore[reportUnknownMemberType]

    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "build_context")
    workflow.add_edge("build_context", generate_node_name)
    workflow.add_edge(generate_node_name, "validate")

    # Conditional edge after validation
    if enable_self_correction:
        def should_retry(state: AgentState) -> str:
            answer = state.get("answer", "") or ""
            retry_count = state.get("metadata", {}).get("retry_count", 0)
            citations_valid = state.get("citations_valid", False)
            if not citations_valid and retry_count < max_retries and answer and "llm_service is required" not in answer:
                return "retry"
            return "finalize"
        workflow.add_conditional_edges(
            "validate",
            should_retry,
            {"retry": "retry_generate", "finalize": "finalize"},
        )
        workflow.add_edge("retry_generate", "validate")
    elif enable_fallback:
        def should_fallback(state: AgentState) -> str:
            citations = state.get("citations", [])
            citations_valid = state.get("citations_valid", True)
            confidence = state.get("confidence", 1.0)
            if not citations:
                return "fallback"
            if not citations_valid or (confidence is not None and confidence < 0.5):
                return "fallback"
            return "finalize"
        workflow.add_conditional_edges(
            "validate",
            should_fallback,
            {"fallback": "web_search_fallback", "finalize": "finalize"},
        )
        workflow.add_edge("web_search_fallback", "finalize")
    else:
        workflow.add_edge("validate", "finalize")

    workflow.add_edge("finalize", END)

    app = workflow.compile(checkpointer=checkpointer)

    try:
        print("RAG Graph Mermaid Syntax:")
        print(app.get_graph().draw_mermaid())
    except Exception:
        logger.warning("Could not print graph visualization", exc_info=True)

    return app


def run_rag_query(
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    retrieval_service: DocumentRetrievalService | None = None,
    reranker: Reranker | None = None,
    llm_service: BaseLLMService | None = None,
    system_prompt: str = "",
    tracer: LangSmithTracer | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    retrieval_config: dict[str, Any] | None = None,
    enable_fallback: bool = False,
    enable_self_correction: bool = False,
    max_retries: int = 2,
) -> RAGResult:
    """Sync entry point — replaces orchestrator.query()."""
    from nagrik_ai.factories import create_retrieval_service

    graph = create_rag_graph(
        retrieval_service=retrieval_service or create_retrieval_service(),
        reranker=reranker,
        llm_service=llm_service,
        system_prompt=system_prompt,
        tracer=tracer,
        checkpointer=checkpointer,
        enable_fallback=enable_fallback,
        enable_self_correction=enable_self_correction,
        max_retries=max_retries,
    )
    initial_state = build_initial_state(
        query, session_id=session_id, user_id=user_id, retrieval_config=retrieval_config
    )
    final_state = graph.invoke(initial_state)
    result = final_state.get("rag_result")
    if result is None:
        raise RuntimeError("RAG graph did not produce a result")
    return result  # type: ignore[no-any-return]


def stream_rag_query(
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    retrieval_service: DocumentRetrievalService | None = None,
    reranker: Reranker | None = None,
    llm_service: BaseLLMService | None = None,
    system_prompt: str = "",
    tracer: LangSmithTracer | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    retrieval_config: dict[str, Any] | None = None,
    enable_fallback: bool = False,
    enable_self_correction: bool = False,
    max_retries: int = 2,
) -> Iterator[dict[str, Any]]:
    """Streaming entry point — replaces orchestrator.query_stream()."""
    from nagrik_ai.factories import create_retrieval_service

    graph = create_rag_graph(
        retrieval_service=retrieval_service or create_retrieval_service(),
        reranker=reranker,
        llm_service=llm_service,
        system_prompt=system_prompt,
        tracer=tracer,
        checkpointer=checkpointer,
        enable_streaming=True,
        enable_fallback=enable_fallback,
        enable_self_correction=enable_self_correction,
        max_retries=max_retries,
    )
    initial_state = build_initial_state(
        query, session_id=session_id, user_id=user_id, retrieval_config=retrieval_config
    )

    # Stream using graph.stream to get node-level updates
    # The generate_stream_node will populate _streaming_buffer with tokens
    # Capture the final state from the last node output (finalize) to avoid a second invoke
    result: RAGResult | None = None
    for chunk in graph.stream(initial_state, stream_mode="updates"):
        node_name = next(iter(chunk.keys()))
        node_output = chunk[node_name]

        if node_name == "generate_stream" and "_streaming_buffer" in node_output:
            for token in node_output["_streaming_buffer"]:
                yield {"type": "token", "content": token}

        if node_name == "finalize" and "rag_result" in node_output:
            result = node_output["rag_result"]

    if result is None:
        raise RuntimeError("RAG graph did not produce a result")
    yield {"type": "final", "data": result}

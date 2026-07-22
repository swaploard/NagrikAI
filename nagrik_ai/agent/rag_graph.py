from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nagrik_ai.agent.nodes import (
    build_context_node,
    generate_node,
    rerank_node,
    retrieve_node,
    validate_node,
)
from nagrik_ai.models.agent_state import AgentState
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.llm_service import BaseLLMService
from nagrik_ai.services.reranker import Reranker

logger = logging.getLogger(__name__)


def create_rag_graph(
    retrieval_service: DocumentRetrievalService,
    reranker: Reranker | None = None,
    llm_service: BaseLLMService | None = None,
    system_prompt: str = "",
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    workflow: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(
        state_schema=AgentState,
        context_schema=None,
    )

    def wrapped_retrieve(state: AgentState) -> dict[str, Any]:
        return retrieve_node(state, retrieval_service)

    def wrapped_rerank(state: AgentState) -> dict[str, Any]:
        return rerank_node(state, reranker)

    def wrapped_generate(state: AgentState) -> dict[str, Any]:
        return generate_node(state, llm_service, system_prompt)

    workflow.add_node("retrieve", wrapped_retrieve)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("rerank", wrapped_rerank)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("build_context", build_context_node)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("generate", wrapped_generate)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("validate", validate_node)  # pyright: ignore[reportUnknownMemberType]

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "build_context")
    workflow.add_edge("build_context", "generate")
    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", END)

    app = workflow.compile(checkpointer=checkpointer)

    try:
        print("RAG Graph Mermaid Syntax:")
        print(app.get_graph().draw_mermaid())
    except Exception:
        logger.warning("Could not print graph visualization", exc_info=True)

    return app

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.state import CompiledStateGraph  # pyright: ignore[reportMissingTypeStubs]

from nagrik_ai.agent.agent_nodes import (
    decide_tool_node,
    execute_tool_node,
    fallback_web_search_node,
    synthesize_node,
)
from nagrik_ai.models.agent_state import AgentState
from nagrik_ai.services.llm_service import BaseLLMService

logger = logging.getLogger(__name__)


def _needs_fallback(state: AgentState) -> bool:
    import re

    tool_results = state.get("tool_results", [])
    for result in tool_results:
        output = result.get("output", "")
        if not isinstance(output, str):
            continue
        if not output.strip():
            return True
        if re.search(
            r"(?:could not|cannot|couldn't|unable to|no information|not found|not available)",
            output,
            re.IGNORECASE,
        ):
            return True
        if "error" in result:
            return True
    return False


def create_agent_graph(
    llm_service: BaseLLMService,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    workflow: StateGraph[AgentState] = StateGraph(AgentState)

    def wrapped_decide_tool(state: AgentState) -> dict[str, Any]:
        return decide_tool_node(state, llm_service)

    def wrapped_synthesize(state: AgentState) -> dict[str, Any]:
        return synthesize_node(state, llm_service)

    def wrapped_fallback(state: AgentState) -> dict[str, Any]:
        return fallback_web_search_node(state, llm_service)

    workflow.add_node("decide_tool", wrapped_decide_tool)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("execute_tool", execute_tool_node)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("synthesize", wrapped_synthesize)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("fallback_web_search", wrapped_fallback)  # pyright: ignore[reportUnknownMemberType]

    workflow.set_entry_point("decide_tool")

    def decide_tool_route(state: AgentState) -> Literal["execute_tool", "synthesize"]:
        return "execute_tool" if state.get("current_tool") else "synthesize"

    def execute_tool_route(state: AgentState) -> Literal["synthesize", "fallback_web_search"]:
        return "fallback_web_search" if _needs_fallback(state) else "synthesize"

    workflow.add_conditional_edges(
        "decide_tool",
        decide_tool_route,
        {"execute_tool": "execute_tool", "synthesize": "synthesize"},
    )

    workflow.add_conditional_edges(
        "execute_tool",
        execute_tool_route,
        {"synthesize": "synthesize", "fallback_web_search": "fallback_web_search"},
    )

    workflow.add_edge("synthesize", END)
    workflow.add_edge("fallback_web_search", END)

    app = workflow.compile(checkpointer=checkpointer)

    try:
        print("Agent Graph Mermaid Syntax:")
        print(app.get_graph().draw_mermaid())
    except Exception:
        logger.warning("Could not print graph visualization", exc_info=True)

    return app

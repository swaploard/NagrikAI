from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nagrik_ai.agent.react_nodes import (
    BusinessProfileReader,
    budget_available,
    execute_tool_node,
    finalize_with_limitations_node,
    initialize_node,
    reason_node,
    synthesize_node,
)
from nagrik_ai.agent.tool_policy import ToolSelectionPolicy
from nagrik_ai.agent.validation_nodes import validate_claims_node, validate_evidence_node
from nagrik_ai.config.config_models import MAX_ITERATIONS, MAX_VALIDATION_RETRIES
from nagrik_ai.models.agent_state import AgentState
from nagrik_ai.services.llm_service import BaseLLMService


def create_agent_graph(
    llm_service: BaseLLMService,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    business_profile_service: BusinessProfileReader | None = None,
    tool_policy: ToolSelectionPolicy | None = None,
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    policy = tool_policy or ToolSelectionPolicy()
    workflow: StateGraph[AgentState] = StateGraph(AgentState)

    def initialize(state: AgentState) -> dict[str, Any]:
        return initialize_node(state, policy, business_profile_service)

    def reason(state: AgentState) -> dict[str, Any]:
        return reason_node(state, llm_service, policy)

    def execute(state: AgentState) -> dict[str, Any]:
        return execute_tool_node(state, policy)

    def validate_evidence(state: AgentState) -> dict[str, Any]:
        return validate_evidence_node(state, policy)

    def synthesize(state: AgentState) -> dict[str, Any]:
        return synthesize_node(state, llm_service)

    def reason_route(state: AgentState) -> str:
        return "execute" if state.get("tool_calls") else "validate_evidence"

    workflow.add_node("initialize", initialize)
    workflow.add_node("reason", reason)
    workflow.add_node("execute", execute)
    workflow.add_node("validate_evidence", validate_evidence)
    workflow.add_node("synthesize", synthesize)
    workflow.add_node("validate_claims", validate_claims_node)
    workflow.add_node("finalize_with_limitations", finalize_with_limitations_node)
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "reason")
    workflow.add_conditional_edges(
        "reason",
        reason_route,
        ["execute", "validate_evidence"],
    )
    workflow.add_edge("execute", "reason")

    def validation_route(state: AgentState, success: str) -> str:
        if not state.get("validation_errors"):
            return success
        if state.get("validation_retries", 0) < state.get("max_validation_retries", 0) and budget_available(state):
            return "reason"
        return "finalize_with_limitations"

    def evidence_route(state: AgentState) -> str:
        return validation_route(state, "synthesize")

    def claims_route(state: AgentState) -> str:
        return validation_route(state, END)

    workflow.add_conditional_edges(
        "validate_evidence",
        evidence_route,
        ["synthesize", "reason", "finalize_with_limitations"],
    )
    workflow.add_edge("synthesize", "validate_claims")
    workflow.add_conditional_edges("validate_claims", claims_route, [END, "reason", "finalize_with_limitations"])
    workflow.add_edge("finalize_with_limitations", END)
    # Allow configured budgets to finish through our terminal node before LangGraph's safety limit.
    return workflow.compile(checkpointer=checkpointer).with_config(
        recursion_limit=max(25, 2 * MAX_ITERATIONS + 4 * MAX_VALIDATION_RETRIES + 6)
    )

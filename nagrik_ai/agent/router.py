from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from nagrik_ai.models.agent_state import AgentStep
from nagrik_ai.models.tool_result import ToolResult
from nagrik_ai.prompts.prompt_loader import load_prompt
from nagrik_ai.services.llm_service import BaseLLMService, create_llm_service
from nagrik_ai.tools.calculator import calculator
from nagrik_ai.tools.compliance_calendar import compliance_calendar
from nagrik_ai.tools.pdf_reader import read_pdf
from nagrik_ai.tools.rag_tool import rag_search_with_sources
from nagrik_ai.tools.scheme_discovery import scheme_discovery
from nagrik_ai.tools.tax_calculator import tax_calculator
from nagrik_ai.tools.web_search import web_search

logger = logging.getLogger(__name__)

TOOL_REGISTRY: dict[str, Any] = {
    "rag_search": rag_search_with_sources,
    "web_search": web_search,
    "read_pdf": read_pdf,
    "calculator": calculator,
    "tax_calculator": tax_calculator,
    "compliance_calendar": compliance_calendar,
    "scheme_discovery": scheme_discovery,
}

AGENT_SYSTEM_PROMPT = load_prompt("agent_system_prompt")


def load_tool_schemas() -> list[dict[str, Any]]:
    schemas_path = Path(__file__).resolve().parent.parent / "tools" / "schemas.json"
    with open(schemas_path) as f:
        data = json.load(f)

    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in data["tools"]
    ]


_checkpointer: Any = None


def _get_checkpointer() -> Any:
    global _checkpointer
    if _checkpointer is None:
        from nagrik_ai.factories import create_checkpointer

        _checkpointer = create_checkpointer()
    return _checkpointer


_ACTION_LABELS: dict[str, str] = {
    "rag_search": "Searching GST guidance…",
    "web_search": "Searching current information…",
    "read_pdf": "Reading document…",
    "calculator": "Calculating liability…",
    "tax_calculator": "Calculating liability…",
    "compliance_calendar": "Checking deadlines…",
    "scheme_discovery": "Finding relevant schemes…",
    "reason": "Choosing next action…",
    "synthesize": "Preparing answer…",
    "validate_evidence": "Checking evidence…",
    "validate_claims": "Checking claims…",
    "finalize_with_limitations": "Disclosing validation limits…",
}


def public_agent_step(step: AgentStep) -> AgentStep:
    """Project internal observations to UI-safe progress, with no arguments or raw errors."""
    result = step.get("tool_result")
    action = step["action"] if step["action"] in _ACTION_LABELS else "action"
    summary = "Searching fallback…" if result is not None and not result.ok else _ACTION_LABELS.get(action, "Working…")
    public_result: ToolResult | None = None
    if result is not None:
        public_result = replace(result, data=None, error_message=None)
    return {
        "action": action,
        "action_input": {},
        "observation_summary": summary,
        "timestamp_ms": step.get("timestamp_ms"),
        "latency_ms": step.get("latency_ms"),
        "tool_result": public_result,
    }


def stream_agent_from_graph(
    graph: Any, query: str, thread_id: str | None = None, *, user_id: str | None = None
) -> Iterator[dict[str, Any]]:
    """Emit new steps once per turn, then the final answer after validation.

    Node updates provide the same incremental buffering boundary as the RAG
    streaming path. Draft answers and private conversation messages never leave it.
    """
    tid = thread_id or uuid4().hex
    state: dict[str, Any] = {}
    seen = 0
    for update in graph.stream(
        {"query": query, "user_id": user_id, "business_context": {}},
        config={"configurable": {"thread_id": tid}},
        stream_mode="updates",
    ):
        for node, output in update.items():
            if not isinstance(output, dict):
                continue
            state.update(output)
            if node == "initialize":
                seen = 0
            steps = output.get("agent_steps", [])
            for step in steps[seen:]:
                yield {"type": "agent_step", "data": public_agent_step(step), "thread_id": tid}
            if "agent_steps" in output:
                seen = len(steps)
    yield {
        "type": "final",
        "thread_id": tid,
        "data": {
            "answer": state.get("answer") or "",
            "citations": state.get("citations", []),
            "citations_valid": state.get("citations_valid"),
            "claim_verdicts": state.get("claim_verdicts", []),
            "finalized_with_limitations": state.get("finalized_with_limitations", False),
        },
    }


def run_agent_from_graph(
    graph: Any,
    query: str,
    thread_id: str | None = None,
    *,
    user_id: str | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, str]:
    answer, tid = "", thread_id or uuid4().hex
    for event in stream_agent_from_graph(graph, query, tid, user_id=user_id):
        if on_event is not None:
            on_event(event)
        if event["type"] == "final":
            answer = event["data"]["answer"]
    return answer, tid


def run_agent(
    query: str,
    llm_service: BaseLLMService | None = None,
    thread_id: str | None = None,
    *,
    user_id: str | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, str]:
    """Run the agent, optionally receiving safe progress events, and retain its thread."""
    from nagrik_ai.factories import create_agent_graph

    graph = create_agent_graph(llm_service=llm_service or create_llm_service(), checkpointer=_get_checkpointer())
    return run_agent_from_graph(graph, query, thread_id, user_id=user_id, on_event=on_event)

"""Bounded orchestration nodes; observations contain results, never private reasoning."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from typing import Any, Protocol, cast
from uuid import uuid4

from httpx import TimeoutException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from requests.exceptions import Timeout as RequestsTimeout

from nagrik_ai.agent.agent_nodes import _base_messages_to_dicts
from nagrik_ai.agent.router import AGENT_SYSTEM_PROMPT, TOOL_REGISTRY, load_tool_schemas
from nagrik_ai.agent.tool_policy import ToolSelectionPolicy
from nagrik_ai.config.config_models import (
    MAX_ITERATIONS,
    MAX_OBSERVATION_CHARS,
    MAX_TOOL_CALLS,
    MAX_VALIDATION_RETRIES,
)
from nagrik_ai.models.agent_state import AgentState, AgentStep
from nagrik_ai.models.tool_result import ErrorCode, SourceAuthority, ToolResult
from nagrik_ai.services.citation_service import bind_tool_citations
from nagrik_ai.services.llm_service import BaseLLMService, RateLimitError


class BusinessProfileReader(Protocol):
    """Read-only injection point for the Phase 4 profile service."""

    def get_profile(self, user_id: str) -> Any: ...


def step(
    action: str, summary: str, result: ToolResult | None = None, arguments: dict[str, Any] | None = None
) -> AgentStep:
    return {
        "action": action,
        "action_input": arguments or {},
        "observation_summary": summary[:MAX_OBSERVATION_CHARS],
        "timestamp_ms": time.time() * 1000,
        "latency_ms": result.latency_ms if result else None,
        "tool_result": result,
    }


def failure(message: str, code: ErrorCode = "VALIDATION_ERROR") -> ToolResult:
    return ToolResult(False, None, code, message, None, None)


def initialize_node(
    state: AgentState, policy: ToolSelectionPolicy, profile_service: BusinessProfileReader | None = None
) -> dict[str, Any]:
    messages = list(state.get("messages", []))
    completed = state.get("metadata", {}).get("agent_turn_completed", False)
    if not messages or not isinstance(messages[-1], HumanMessage) or messages[-1].content != state["query"]:
        messages.append(HumanMessage(content=state["query"]))
    context = dict(state.get("business_context", {}))
    if profile_service is not None and state.get("user_id"):
        profile = profile_service.get_profile(cast(str, state["user_id"]))
        if profile is not None:
            context = dict(profile.model_dump() if hasattr(profile, "model_dump") else profile)
    evidence_policy = None if completed else state.get("evidence_policy")
    return {
        "messages": messages,
        "business_context": context,
        "evidence_policy": evidence_policy
        or policy.resolve_evidence_policy(
            state["query"], None if completed else state.get("metadata", {}).get("question_type")
        ),
        "iteration": 0,
        "tool_calls_count": 0,
        "validation_retries": 0,
        "max_iterations": max(0, state.get("max_iterations", MAX_ITERATIONS)),
        "max_tool_calls": max(0, state.get("max_tool_calls", MAX_TOOL_CALLS)),
        "max_validation_retries": max(0, state.get("max_validation_retries", MAX_VALIDATION_RETRIES)),
        "agent_steps": [],
        "claim_verdicts": [],
        "validation_errors": [],
        "missing_info": [],
        "tool_results": [],
        "tool_calls": [],
        "current_tool": None,
        "answer": None,
        "confidence": None,
        "citations": [],
        "citations_valid": None,
        "documents": [],
        "context": None,
        "rag_result": None,
        "finalized_with_limitations": False,
        "pending_profile_updates": None,
        "metadata": {**state.get("metadata", {}), "agent_turn_completed": False},
    }


def budget_available(state: AgentState) -> bool:
    return state.get("iteration", 0) < state.get("max_iterations", MAX_ITERATIONS) and state.get(
        "tool_calls_count", 0
    ) < state.get("max_tool_calls", MAX_TOOL_CALLS)


def _system_prompt(state: AgentState) -> str:
    evidence = state.get("evidence_policy")
    context = {
        "business_context": state.get("business_context", {}),
        "evidence_policy": asdict(evidence) if evidence else None,
        "observations": [s["observation_summary"] for s in state.get("agent_steps", [])],
        "validation_errors": state.get("validation_errors", []),
        "missing_info": state.get("missing_info", []),
        "citation_sources": state.get("citations", []),
    }
    return (
        AGENT_SYSTEM_PROMPT
        + "\nRuntime context:\n"
        + json.dumps(
            context, default=lambda value: sorted(value) if isinstance(value, (set, frozenset)) else str(value)
        )
    )


def reason_node(
    state: AgentState, llm: BaseLLMService, tool_policy: ToolSelectionPolicy | None = None
) -> dict[str, Any]:
    policy = tool_policy or ToolSelectionPolicy()
    if not budget_available(state):
        return {"tool_calls": [], "current_tool": None}
    evidence = state.get("evidence_policy")
    response = llm.chat(
        _base_messages_to_dicts(state.get("messages", [])),
        tools=load_tool_schemas(),
        system=_system_prompt(state),
    )
    calls = []
    for tc in response.tool_calls or []:
        calls.append(
            {
                "name": tc.name,
                "arguments": tc.arguments,
                "id": tc.id or uuid4().hex,
                "policy_error": policy.validate_tool_call(tc.name, evidence),
            }
        )
    # Tool-selection prose may contain private reasoning. Only retain the calls.
    message = AIMessage(
        content="" if calls else (response.content or ""),
        tool_calls=[{"name": tc["name"], "args": tc["arguments"], "id": tc["id"]} for tc in calls],
    )
    return {
        "tool_calls": calls,
        "current_tool": calls[0]["name"] if calls else None,
        "messages": [*state.get("messages", []), message],
        "answer": None if calls else response.content,
        "agent_steps": [*state.get("agent_steps", []), step("reason", "Selected next action.")],
    }


def _normalize(raw: Any, name: str, call_id: str) -> ToolResult:
    if isinstance(raw, ToolResult):
        return raw
    if (
        raw is None
        or raw == ""
        or raw == []
        or (
            isinstance(raw, str)
            and raw in {"No search results found.", "No content found in search results.", "PDF contains no pages."}
        )
    ):
        return failure("No evidence found; select an alternative permitted tool.", "NOT_FOUND")
    if name == "rag_search":
        if not isinstance(raw, dict) or not raw.get("sources") or not raw.get("citations_valid"):
            return failure("No validated retrieval evidence found.", "NOT_FOUND")
        ids = tuple(str(s["source_id"]) for s in raw["sources"] if s.get("source_id"))
        return ToolResult(True, raw, None, None, "authoritative", None, ids)
    if name == "calculator":
        return ToolResult(True, str(raw), None, None, "deterministic", None, (f"calculator:{call_id}",))
    authority: SourceAuthority = "general_web" if name == "web_search" else "secondary"
    source_ids = (f"{name}:{call_id}",)
    if name == "read_pdf" and isinstance(raw, dict):
        document = raw.get("document", raw)
        authority = document.get("source_authority", "secondary")
        if document.get("source_id"):
            source_ids = (str(document["source_id"]),)
        if authority not in {"authoritative", "secondary", "general_web"}:
            return failure("Invalid PDF document authority.")
    return ToolResult(
        True,
        raw if isinstance(raw, (str, dict, list)) else str(raw),
        None,
        None,
        authority,
        None,
        source_ids,
    )


def execute_tool_node(state: AgentState, tool_policy: ToolSelectionPolicy | None = None) -> dict[str, Any]:
    policy = tool_policy or ToolSelectionPolicy()
    steps = list(state.get("agent_steps", []))
    results = list(state.get("tool_results", []))
    messages = list(state.get("messages", []))
    count = state.get("tool_calls_count", 0)
    citations = list(state.get("citations", []))
    schemas = {s["function"]["name"]: s["function"]["parameters"] for s in load_tool_schemas()}
    previous = next((s["action"] for s in reversed(steps) if s["tool_result"] is not None), None)
    for tc in state.get("tool_calls", []):
        name, arguments = tc["name"], tc["arguments"]
        start = time.perf_counter()
        result = tc.get("policy_error") or policy.validate_tool_call(name, state.get("evidence_policy"))
        if count >= state.get("max_tool_calls", MAX_TOOL_CALLS) or state.get("iteration", 0) >= state.get(
            "max_iterations", MAX_ITERATIONS
        ):
            result = failure("Tool call budget exhausted.")
        else:
            count += 1  # Every attempted call, including denied calls, consumes the call budget.
            if policy.check_forbidden_transition(previous, name, state.get("evidence_policy")):
                result = failure("Forbidden tool transition.", "FORBIDDEN_TRANSITION")
            if result is None:
                schema = schemas.get(name)
                if name not in TOOL_REGISTRY or schema is None:
                    result = failure(f"Unknown tool: {name}")
                elif (
                    not isinstance(arguments, dict)
                    or set(schema.get("required", [])) - arguments.keys()
                    or arguments.keys() - schema.get("properties", {}).keys()
                    or any(not isinstance(v, str) for v in arguments.values())
                ):
                    result = failure(f"Invalid arguments for {name}.")
                else:
                    try:
                        result = _normalize(TOOL_REGISTRY[name](**arguments), name, tc["id"])
                    except (TimeoutError, TimeoutException, RequestsTimeout):
                        result = failure("Tool timed out; retry within the remaining budget.", "TIMEOUT")
                    except FileNotFoundError:
                        result = failure("Evidence not found; use an alternative permitted tool.", "NOT_FOUND")
                    except RateLimitError:
                        result = failure("Tool rate limited.", "RATE_LIMITED")
                    except Exception as exc:
                        cause = exc.__cause__ or exc
                        if isinstance(cause, (TimeoutError, TimeoutException, RequestsTimeout)):
                            result = failure("Tool timed out; retry within the remaining budget.", "TIMEOUT")
                        elif getattr(getattr(cause, "response", None), "status_code", None) == 429:
                            result = failure("Tool rate limited.", "RATE_LIMITED")
                        else:
                            result = failure(f"Tool failed ({type(exc).__name__}).")
        result = replace(result, latency_ms=(time.perf_counter() - start) * 1000)
        result, citations = bind_tool_citations(result, citations)
        summary = json.dumps(asdict(result), default=str)[:MAX_OBSERVATION_CHARS]
        steps.append(step(name, summary, result, arguments))
        results.append({"tool_call_id": tc["id"], "name": name, "output": result})
        messages.append(ToolMessage(content=summary, tool_call_id=tc["id"]))
        previous = name
    return {
        "tool_results": results,
        "citations": citations,
        "messages": messages,
        "agent_steps": steps,
        "tool_calls_count": count,
        "iteration": state.get("iteration", 0) + 1,
        "tool_calls": [],
        "current_tool": None,
    }


def synthesize_node(state: AgentState, llm: BaseLLMService) -> dict[str, Any]:
    answer = state.get("answer")
    messages = list(state.get("messages", []))
    if not answer:
        response = llm.chat(_base_messages_to_dicts(messages), tools=None, system=_system_prompt(state))
        answer = response.content or "I could not produce a supported answer."
        messages.append(AIMessage(content=answer))
    return {
        "answer": answer,
        "messages": messages,
        "agent_steps": [*state.get("agent_steps", []), step("synthesize", "Prepared answer from observations.")],
    }


def finalize_with_limitations_node(state: AgentState) -> dict[str, Any]:
    unresolved = [v["claim"] for v in state.get("claim_verdicts", []) if v["status"] != "supported"]
    limitations = unresolved + state.get("validation_errors", [])
    disclosure = "> ⚠️ This answer could not be fully validated against authoritative sources. Claims: "
    answer = disclosure + json.dumps(list(dict.fromkeys(limitations)), ensure_ascii=False)
    answer += "\n\n" + (state.get("answer") or "Please provide additional evidence or clarify the missing information.")
    updates: dict[str, Any] = {
        "answer": answer,
        "finalized_with_limitations": True,
        "confidence": min(cast(float, state["confidence"]) if state.get("confidence") is not None else 0.4, 0.4),
        "messages": [*state.get("messages", []), AIMessage(content=answer)],
        "metadata": {**state.get("metadata", {}), "agent_turn_completed": True},
        "agent_steps": [
            *state.get("agent_steps", []),
            step("finalize_with_limitations", "Disclosed validation limits."),
        ],
    }
    if any(v["status"] == "unsupported" for v in state.get("claim_verdicts", [])):
        updates["citations_valid"] = False
    return updates

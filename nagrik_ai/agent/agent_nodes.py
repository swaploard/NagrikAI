from __future__ import annotations

import json
import logging
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from nagrik_ai.agent.router import AGENT_SYSTEM_PROMPT, TOOL_REGISTRY, load_tool_schemas
from nagrik_ai.models.agent_state import AgentState
from nagrik_ai.services.llm_service import BaseLLMService
from nagrik_ai.tools.web_search import web_search

logger = logging.getLogger(__name__)


def _base_messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": cast(str, msg.content)}
            if msg.tool_calls:
                openai_tool_calls: list[dict[str, Any]] = []
                for tc in msg.tool_calls:
                    openai_tool_calls.append(
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": (
                                    json.dumps(tc["args"])
                                    if isinstance(tc["args"], dict)
                                    else str(tc["args"])
                                ),
                            },
                        }
                    )
                entry["tool_calls"] = openai_tool_calls
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            result.append({"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id})
        else:
            result.append({"role": msg.type, "content": msg.content or ""})
    return result


def decide_tool_node(state: AgentState, llm_service: BaseLLMService) -> dict[str, Any]:
    messages = state.get("messages", [])
    query = state["query"]

    msg_dicts = _base_messages_to_dicts(messages) if messages else []
    updated_messages = list(messages)

    if not msg_dicts or msg_dicts[-1].get("role") != "user":
        msg_dicts.append({"role": "user", "content": query})
        if not updated_messages or not isinstance(updated_messages[-1], HumanMessage):
            updated_messages.append(HumanMessage(content=query))

    tool_schemas = load_tool_schemas()
    response = llm_service.chat(msg_dicts, tools=tool_schemas, system=AGENT_SYSTEM_PROMPT)

    tool_calls_list: list[dict[str, Any]] = []
    current_tool: str | None = None

    if response.tool_calls:
        tool_calls_list = [
            {"name": tc.name, "arguments": tc.arguments, "id": tc.id}
            for tc in response.tool_calls
        ]
        current_tool = response.tool_calls[0].name if response.tool_calls else None

    ai_tool_calls = [
        {"name": tc.name, "args": tc.arguments, "id": tc.id, "type": "tool_call"}
        for tc in (response.tool_calls or [])
    ]

    ai_message = AIMessage(content=response.content or "", tool_calls=ai_tool_calls)

    return {
        "tool_calls": tool_calls_list,
        "current_tool": current_tool,
        "messages": [*updated_messages, ai_message],
    }


def execute_tool_node(state: AgentState) -> dict[str, Any]:
    tool_calls = state.get("tool_calls", [])
    tool_results: list[dict[str, Any]] = []
    tool_messages: list[ToolMessage] = []

    for tc in tool_calls:
        tool_fn = TOOL_REGISTRY.get(tc["name"])
        if tool_fn is None:
            logger.warning("Unknown tool: %s", tc["name"])
            result = f"Unknown tool: {tc['name']}"
        else:
            try:
                logger.info("Executing tool: %s with args: %s", tc["name"], tc["arguments"])
                result = tool_fn(**tc["arguments"])
                if not isinstance(result, str):
                    result = str(result)
            except Exception as e:
                logger.error("Tool %s failed: %s", tc["name"], e)
                result = f"Error executing {tc['name']}: {e}"

        tool_results.append({"tool_call_id": tc.get("id", ""), "output": result})
        tool_messages.append(ToolMessage(content=result, tool_call_id=tc.get("id", "")))

    return {
        "tool_results": tool_results,
        "messages": state.get("messages", []) + tool_messages,
    }


def synthesize_node(state: AgentState, llm_service: BaseLLMService) -> dict[str, Any]:
    tool_results = state.get("tool_results", [])
    messages = state.get("messages", [])

    if tool_results:
        msg_dicts = _base_messages_to_dicts(messages)
        response = llm_service.chat(msg_dicts, tools=None, system=AGENT_SYSTEM_PROMPT)
        ai_message = AIMessage(content=response.content or "")
        return {
            "answer": response.content or "",
            "messages": [*messages, ai_message],
        }

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return {"answer": msg.content}

    return {"answer": "I could not generate a response."}


def fallback_web_search_node(state: AgentState, llm_service: BaseLLMService) -> dict[str, Any]:
    query = state["query"]
    logger.info("RAG returned no useful answer; directly invoking web_search fallback")

    web_result = web_search(query)

    synthesis_messages: list[dict[str, Any]] = [
        {"role": "user", "content": query},
        {
            "role": "user",
            "content": (
                f"Web search results:\n{web_result}\n\n"
                f"Please provide a comprehensive answer based on these search results."
            ),
        },
    ]

    response = llm_service.chat(synthesis_messages, tools=None, system=AGENT_SYSTEM_PROMPT)

    return {
        "answer": response.content or "",
        "messages": [
            HumanMessage(content=query),
            HumanMessage(
                content=(
                    f"Web search results:\n{web_result}\n\n"
                    f"Please provide a comprehensive answer based on these search results."
                )
            ),
            AIMessage(content=response.content or ""),
        ],
    }

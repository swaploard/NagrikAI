from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage

from nagrik_ai.prompts.prompt_loader import load_prompt
from nagrik_ai.services.llm_service import BaseLLMService, create_llm_service
from nagrik_ai.tools.pdf_reader import read_pdf
from nagrik_ai.tools.rag_tool import rag_search
from nagrik_ai.tools.web_search import web_search

logger = logging.getLogger(__name__)

TOOL_REGISTRY: dict[str, Any] = {
    "rag_search": rag_search,
    "web_search": web_search,
    "read_pdf": read_pdf,
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


def run_agent(
    query: str,
    llm_service: BaseLLMService | None = None,
    thread_id: str | None = None,
) -> tuple[str, str]:
    """Run the agent graph: decide, execute tool(s), synthesize answer.

    Args:
        query: The user's question.
        llm_service: Optional LLM service override.
        thread_id: Optional conversation thread ID for multi-turn. If None,
                   a new thread is created.

    Returns:
        Tuple of (answer, thread_id). Use thread_id in subsequent calls to
        continue the same conversation.
    """
    logger.info("Agent query: %s", query)

    if llm_service is None:
        llm_service = create_llm_service()

    from nagrik_ai.factories import create_agent_graph
    from nagrik_ai.models.agent_state import AgentState

    checkpointer = _get_checkpointer()
    agent_graph = create_agent_graph(llm_service=llm_service, checkpointer=checkpointer)

    tid = thread_id or uuid4().hex

    if thread_id is None:
        initial_state: AgentState = {
            "query": query,
            "rewritten_queries": [],
            "documents": [],
            "candidate_answers": [],
            "answer": None,
            "confidence": None,
            "citations": [],
            "errors": [],
            "metadata": {},
            "tool_calls": [],
            "tool_results": [],
            "current_tool": None,
            "session_id": None,
            "user_id": None,
            "trace_id": None,
            "context": None,
            "retrieval_config": {},
            "citations_valid": None,
            "rag_result": None,
            "_streaming_buffer": None,
            "_streaming_callback": None,
            "messages": [HumanMessage(content=query)],
        }
        final_state = agent_graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": tid}},
        )
    else:
        final_state = agent_graph.invoke(
            {"query": query},
            config={"configurable": {"thread_id": tid}},
        )

    return final_state.get("answer", "") or "", tid

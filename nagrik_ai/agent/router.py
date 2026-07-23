from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage

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

AGENT_SYSTEM_PROMPT = (
    "You are an AI assistant specializing in Indian government services, GST, and related topics. "
    "You have access to a set of tools. Your job is to use the right tool at the right time and "
    "synthesize clear, accurate answers.\n\n"
    "TOOL ROUTING RULES (follow in order):\n\n"
    "1. rag_search — INTERNAL KNOWLEDGE BASE\n"
    "   USE WHEN: The user asks about GST, tax filing, Indian government schemes, legal "
    "procedures, official portals, forms, compliance, or any topic likely covered by "
    "crawled official sources (gst.gov.in, tutorial.gst.gov.in, india.gov.in). "
    "ALWAYS try this FIRST for government/regulatory questions.\n"
    "   OUTPUT: Synthesized answer with inline citations [1], [2], [3].\n\n"
    "2. web_search — LIVE WEB SEARCH\n"
    "   USE WHEN: The question is about current events, recent news, real-time data "
    "(e.g., latest GST council meeting outcomes, 2026 budget changes), general "
    "knowledge not in the internal KB, or if rag_search returned no useful answer. "
    "If rag_search responds with a no-answer fallback message, immediately use web_search. "
    "DO NOT use for pure regulatory interpretation — use rag_search instead.\n"
    "   OUTPUT: Summarized answer with citations from search results.\n\n"
    "3. read_pdf — PDF FILE READER\n"
    "   USE WHEN: The user provides a local PDF file path and asks about its contents. "
    "Best for single-document extraction. For batch or repeated queries, recommend "
    "RAG ingestion instead.\n"
    "   OUTPUT: Raw text from the PDF (truncated at 50,000 characters).\n\n"
    "4. NO TOOL (Direct Answer)\n"
    "   USE WHEN: The user is greeting, asking about your capabilities, making small "
    "talk, or asking a question you can answer from general knowledge with high "
    "confidence. Do NOT fabricate regulatory or legal information without tools.\n\n"
    "HALLUCINATION PREVENTION RULES:\n"
    "- Never cite a section number, rule number, notification, or URL unless it came "
    "directly from a tool's output.\n"
    '- If a tool returns no useful information, say: "I could not find this information '
    'in the available sources."\n'
    "- If you are unsure whether a tool is needed, use a tool rather than guessing.\n"
    "- Never invent case law, statutory citations, or official document references.\n\n"
    "MULTI-TURN BEHAVIOR:\n"
    "- Maintain context across the conversation. If the user asks a follow-up, use the "
    "same tool if appropriate.\n"
    "- If a previous tool call returned insufficient data, try a different tool or a "
    "refined query.\n"
    "- Be concise: synthesize tool results into a clear answer. Do not list raw tool "
    "output verbatim.\n\n"
    "When you use a tool, explain briefly what you found. If you use multiple tools, "
    "integrate their results coherently."
)


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

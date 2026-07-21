from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
    "- If a tool returns no useful information, say: \"I could not find this information "
    "in the available sources.\"\n"
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


def run_agent(query: str, llm_service: BaseLLMService | None = None) -> str:
    """Run the agent loop: decide, execute tool(s), synthesize answer."""
    logger.info("Agent query: %s", query)

    if llm_service is None:
        llm_service = create_llm_service()

    tool_schemas = load_tool_schemas()

    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
    response = llm_service.chat(messages, tools=tool_schemas, system=AGENT_SYSTEM_PROMPT)

    max_tool_rounds = 3
    for _ in range(max_tool_rounds):
        if not response.tool_calls:
            break

        # Add assistant's message to conversation history
        assistant_message: dict[str, Any] = {"role": "assistant"}
        if response.content is not None:
            assistant_message["content"] = response.content
        if response.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in response.tool_calls
            ]
        messages.append(assistant_message)

        tool_results: list[dict[str, Any]] = []
        for tc in response.tool_calls:
            tool_fn = TOOL_REGISTRY.get(tc.name)
            if tool_fn is None:
                logger.warning("Unknown tool: %s", tc.name)
                result = f"Unknown tool: {tc.name}"
            else:
                try:
                    logger.info("Executing tool: %s with args: %s", tc.name, tc.arguments)
                    result = tool_fn(**tc.arguments)
                    if not isinstance(result, str):
                        result = str(result)
                except Exception as e:
                    logger.error("Tool %s failed: %s", tc.name, e)
                    result = f"Error executing {tc.name}: {e}"

            tool_results.append({"role": "tool", "content": result, "tool_call_id": tc.id})

        messages.extend(tool_results)

        # Check if rag_search returned no useful answer
        rag_no_answer = any(
            isinstance(item.get("content"), str)
            and "I could not find this information" in item.get("content", "")
            and any(tc.name == "rag_search" for tc in response.tool_calls)
            for item in tool_results
        )

        if rag_no_answer:
            logger.info("RAG returned no useful answer; directly invoking web_search fallback")
            # Directly call web_search instead of asking LLM to do it
            web_result = web_search(query)
            # Build a clean message history for synthesis: user query + web result as context
            synthesis_messages = [
                {"role": "user", "content": query},
                {"role": "user", "content": f"Web search results:\n{web_result}\n\nPlease provide a comprehensive answer based on these search results."},
            ]
            # Let LLM synthesize final answer from web results (no tools needed)
            response = llm_service.chat(synthesis_messages, tools=None, system=AGENT_SYSTEM_PROMPT)
            break

        # Get next response from LLM (only if we didn't break for fallback)
        response = llm_service.chat(messages, tools=tool_schemas, system=AGENT_SYSTEM_PROMPT)

    return response.content or ""

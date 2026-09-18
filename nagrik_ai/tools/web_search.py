"""Tavily search with explicit general-web provenance."""

from __future__ import annotations

import os

import requests

from nagrik_ai.models.tool_result import ToolResult
from nagrik_ai.tools.result_utils import failure, source_id, tool_result


@tool_result
def web_search(query: str) -> ToolResult:
    if not isinstance(query, str) or not query.strip():
        return failure("Search query must not be empty.")
    api_key = os.getenv("NAGRIKAI_TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
    if not api_key:
        return failure("Tavily API key is not configured.")
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "max_results": 5,
        },
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    sources = []
    for item in payload.get("results", [])[:5]:
        if not item.get("url") or not item.get("content"):
            continue
        sources.append(
            {
                "citation_id": len(sources) + 1,
                "source_id": source_id("tavily", item["url"]),
                "url": item["url"],
                "title": item.get("title", ""),
                "content": item["content"],
            }
        )
    if not sources:
        return failure("No sourced web results found.", "NOT_FOUND")
    return ToolResult(
        True,
        {"response": payload.get("answer") or "\n\n".join(s["content"] for s in sources), "sources": sources},
        None,
        None,
        "general_web",
        None,
        tuple(s["source_id"] for s in sources),
    )

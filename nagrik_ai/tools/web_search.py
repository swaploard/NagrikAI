"""Web search tool using Tavily API."""

import os
from typing import Any

import requests


def web_search(query: str) -> str:
    """
    Perform a web search using Tavily API and return summarized results.

    Args:
        query: The search query string.

    Returns:
        A string containing summarized search results.

    Raises:
        ValueError: If TAVILY_API_KEY environment variable is not set.
        requests.RequestException: If the API request fails.
    """
    api_key = os.getenv("NAGRIKAI_TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    url = "https://api.tavily.com/search"
    payload: dict[str, str | bool | int] = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": 5,
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        # Extract the answer if available, otherwise combine snippets
        if data.get("answer"):
            return data["answer"]

        # Fallback: combine content from results
        results = data.get("results", [])
        if not results:
            return "No search results found."

        snippets: list[str] = []
        for result in results[:3]:  # Use top 3 results
            content = result.get("content", "")
            if content:
                snippets.append(content)

        return "\n\n".join(snippets) if snippets else "No content found in search results."

    except requests.RequestException as e:
        raise requests.RequestException(f"Tavily API request failed: {e!s}") from e

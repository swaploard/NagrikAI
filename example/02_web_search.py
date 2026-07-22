"""Example 2: Web Search Routing Tests — Full Integration (Real LLM + Real Tavily).

Verifies the agent routes current-events queries to web_search using
real OpenRouter LLM and real Tavily API. No mocks.
"""

from __future__ import annotations

import pytest

from nagrik_ai.agent.router import run_agent


@pytest.mark.timeout(600)
def test_budget_highlights_routes_to_web_search() -> None:
    """'Latest Indian budget 2026 highlights' → web_search via real Tavily."""
    query = "Latest Indian budget 2026 highlights"
    result, _ = run_agent(query)

    assert result is not None
    assert len(result) > 50
    assert "budget" in result.lower()


@pytest.mark.timeout(600)
def test_inflation_rate_routes_to_web_search() -> None:
    """'Current inflation rate in India' → web_search via real Tavily."""
    query = "Current inflation rate in India"
    result, _ = run_agent(query)

    assert result is not None
    assert len(result) > 50
    assert "inflation" in result.lower()


def test_ipl_winner_routes_to_web_search() -> None:
    """'Who won the last IPL?' → web_search via real Tavily."""
    query = "Who won the last IPL?"
    result, _ = run_agent(query)

    assert result is not None
    assert len(result) > 50

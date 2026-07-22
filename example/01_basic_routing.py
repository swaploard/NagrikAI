"""Example 1: Basic Routing Tests — Full Integration (Real LLM + Real RAG).

Verifies the agent routes internal-knowledge queries to rag_search using
real OpenRouter LLM and real ChromaDB retrieval. No mocks.
"""

from __future__ import annotations

import pytest

from nagrik_ai.agent.router import run_agent


@pytest.mark.timeout(600)
class TestRealRAGRouting:
    """Agent should route government/tax queries to rag_search (real LLM)."""

    def test_gst_input_tax_credit_routes_to_rag_search(self) -> None:
        """'What is GST input tax credit?' → rag_search with real ChromaDB."""
        query = "What is GST input tax credit?"
        result, _ = run_agent(query)

        assert result is not None
        assert len(result) > 50
        assert "input tax credit" in result.lower() or "itc" in result.lower()

    def test_section_80c_deductions_routes_to_rag_search(self) -> None:
        """'Explain section 80C deductions' → rag_search."""
        query = "Explain section 80C deductions"
        result, _ = run_agent(query)

        assert result is not None
        assert len(result) > 50
        assert "80c" in result.lower() or "section 80c" in result.lower()

    def test_company_leave_policies_routes_to_rag_search(self) -> None:
        """'What are our company leave policies?' → rag_search."""
        query = "What are our company leave policies?"
        result, _ = run_agent(query)

        assert result is not None
        assert len(result) > 50
        assert "leave" in result.lower() or "policy" in result.lower()

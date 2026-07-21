from __future__ import annotations

import pytest
from nagrik_ai.agent.router import run_agent
from nagrik_ai.services.llm_service import LLMResponse, ToolCall


class StubLLMService:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.tool_call_count = 0

    def chat(self, messages: list[dict[str, object]], tools: object | None = None, system: str | None = None):
        self.calls.append(messages)
        if self.tool_call_count == 0:
            self.tool_call_count += 1
            return LLMResponse(
                tool_calls=[ToolCall(name="rag_search", arguments={"query": "What is IFF?"}, id="call_1")]
            )
        if self.tool_call_count == 1:
            self.tool_call_count += 1
            # Second call: LLM synthesizes answer from web search results (no tool calls)
            return LLMResponse(content="Final answer from fallback")
        return LLMResponse(content="Final answer from fallback")


def test_run_agent_falls_back_to_web_search_when_rag_has_no_answer(monkeypatch: pytest.MonkeyPatch):
    stub_llm = StubLLMService()
    web_calls: list[str] = []

    def fake_rag_search(query: str, **_: object) -> str:
        return "I could not find this information in the provided sources."

    def fake_web_search(query: str, **_: object) -> str:
        web_calls.append(query)
        return "Web answer"

    # Mock the actual web_search function that gets imported in router.py
    monkeypatch.setattr("nagrik_ai.agent.router.web_search", fake_web_search)
    monkeypatch.setattr("nagrik_ai.agent.router.TOOL_REGISTRY", {
        "rag_search": fake_rag_search,
        "web_search": fake_web_search,  # This won't actually be called in our implementation
        "read_pdf": lambda **_kwargs: ""  # type: ignore[return-value]
    })

    result = run_agent("What is IFF?", llm_service=stub_llm)

    assert result == "Final answer from fallback"
    assert web_calls == ["What is IFF?"]

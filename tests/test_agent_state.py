from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nagrik_ai.models.agent_state import AgentState


def test_agent_state_all_fields() -> None:
    state: AgentState = {
        "query": "What is GST?",
        "rewritten_queries": ["GST meaning", "GST full form"],
        "documents": [{"page_content": "GST is Goods and Services Tax", "source_id": "doc1"}],
        "candidate_answers": ["GST is Goods and Services Tax"],
        "answer": "GST is Goods and Services Tax",
        "confidence": 0.85,
        "citations": [{"citation_id": 1, "title": "GST Guide", "url": "https://example.com/gst"}],
        "errors": [],
        "metadata": {"retrieved_count": 1},
        "tool_calls": [{"name": "rag_search", "arguments": {"query": "GST"}, "id": "call_1"}],
        "tool_results": [{"tool_call_id": "call_1", "output": "GST info"}],
        "current_tool": "rag_search",
        "session_id": "sess_001",
        "user_id": "user_001",
        "trace_id": "trace_001",
        "context": "Some context text",
        "retrieval_config": {"top_k": 5, "reranker_enabled": False},
        "messages": [HumanMessage(content="What is GST?")],
    }
    assert state["query"] == "What is GST?"
    assert state["answer"] == "GST is Goods and Services Tax"
    assert state["confidence"] == 0.85
    assert state["current_tool"] == "rag_search"
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)


def test_agent_state_optional_nulls() -> None:
    state: AgentState = {
        "query": "test",
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
        "messages": [],
    }
    assert state["answer"] is None
    assert state["confidence"] is None
    assert state["current_tool"] is None
    assert state["session_id"] is None
    assert state["context"] is None
    assert state["messages"] == []


def test_agent_state_messages_variants() -> None:
    state: AgentState = {
        "query": "test",
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
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            ToolMessage(content="Result", tool_call_id="call_1"),
        ],
    }
    assert len(state["messages"]) == 3
    assert isinstance(state["messages"][0], HumanMessage)
    assert isinstance(state["messages"][1], AIMessage)
    assert isinstance(state["messages"][2], ToolMessage)


def test_agent_state_required_fields() -> None:
    state: AgentState = {
        "query": "",
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
        "messages": [],
    }
    assert isinstance(state, dict)
    assert "query" in state
    assert "messages" in state

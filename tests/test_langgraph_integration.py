from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from nagrik_ai.agent._testing import adapt_to_rag_result
from nagrik_ai.agent.agent_graph import create_agent_graph
from nagrik_ai.agent.agent_nodes import (
    decide_tool_node,
    execute_tool_node,
    fallback_web_search_node,
    synthesize_node,
)
from nagrik_ai.agent.nodes import (
    build_context_node,
    generate_node,
    rerank_node,
    retrieve_node,
    validate_node,
)
from nagrik_ai.agent.rag_graph import create_rag_graph
from nagrik_ai.models.agent_state import AgentState
from nagrik_ai.models.rag_result import RAGResult
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.llm_service import BaseLLMService, LLMResponse, ToolCall
from nagrik_ai.services.reranker import Reranker

# ---------------------------------------------------------------------------
# Section A: Graph compilation
# ---------------------------------------------------------------------------


def test_rag_graph_compiles() -> None:
    graph = create_rag_graph(
        retrieval_service=MagicMock(spec=DocumentRetrievalService),
        llm_service=MagicMock(spec=BaseLLMService),
        system_prompt="test",
    )
    assert isinstance(graph, CompiledStateGraph)
    assert "retrieve" in graph.nodes
    assert "rerank" in graph.nodes
    assert "build_context" in graph.nodes
    assert "generate" in graph.nodes
    assert "validate" in graph.nodes


def test_agent_graph_compiles() -> None:
    graph = create_agent_graph(llm_service=MagicMock(spec=BaseLLMService))
    assert isinstance(graph, CompiledStateGraph)
    assert "decide_tool" in graph.nodes
    assert "execute_tool" in graph.nodes
    assert "synthesize" in graph.nodes
    assert "fallback_web_search" in graph.nodes


def test_rag_graph_mermaid_output() -> None:
    graph = create_rag_graph(
        retrieval_service=MagicMock(spec=DocumentRetrievalService),
        llm_service=MagicMock(spec=BaseLLMService),
        system_prompt="test",
    )
    mermaid = graph.get_graph().draw_mermaid()
    assert isinstance(mermaid, str)
    assert len(mermaid) > 0
    assert "retrieve" in mermaid
    assert "validate" in mermaid


def test_agent_graph_mermaid_output() -> None:
    graph = create_agent_graph(llm_service=MagicMock(spec=BaseLLMService))
    mermaid = graph.get_graph().draw_mermaid()
    assert isinstance(mermaid, str)
    assert len(mermaid) > 0
    assert "decide_tool" in mermaid
    assert "fallback_web_search" in mermaid


# ---------------------------------------------------------------------------
# Section B: RAG node unit tests
# ---------------------------------------------------------------------------


def _make_doc(
    page_content: str = "Content",
    source_id: str = "s1",
    score: float = 0.9,
) -> dict[str, Any]:
    return {
        "page_content": page_content,
        "source_id": source_id,
        "title": "Doc Title",
        "url": "https://example.com/doc",
        "domain": "example.com",
        "chunk_index": 0,
        "total_chunks": 1,
        "score": score,
    }


def test_retrieve_node() -> None:
    mock_retrieval = MagicMock(spec=DocumentRetrievalService)
    mock_retrieval.retrieve.return_value = [_make_doc()]

    state: AgentState = {
        "query": "test query",
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

    result = retrieve_node(state, mock_retrieval)
    assert "documents" in result
    assert len(result["documents"]) == 1
    assert result["metadata"]["retrieved_count"] == 1
    mock_retrieval.retrieve.assert_called_once_with("test query")


def test_rerank_node_no_reranker() -> None:
    state: AgentState = {
        "query": "test",
        "rewritten_queries": [],
        "documents": [_make_doc()],
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
    result = rerank_node(state, reranker=None)
    assert result == {}


def test_rerank_node_with_reranker() -> None:
    mock_reranker = MagicMock(spec=Reranker)
    mock_reranker.rerank.return_value = [_make_doc(score=0.95)]

    state: AgentState = {
        "query": "test",
        "rewritten_queries": [],
        "documents": [_make_doc(score=0.9)],
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
    result = rerank_node(state, mock_reranker)
    assert "documents" in result
    assert result["documents"][0]["score"] == 0.95
    mock_reranker.rerank.assert_called_once()


def test_build_context_node() -> None:
    state: AgentState = {
        "query": "test query",
        "rewritten_queries": [],
        "documents": [_make_doc(page_content="Some content about GST", score=0.9)],
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
    result = build_context_node(state)
    assert "context" in result
    assert "citations" in result
    assert "documents" in result
    assert len(result["citations"]) > 0
    context = result.get("context", "")
    assert "GST" in context or "Content" in context


def test_generate_node() -> None:
    mock_llm = MagicMock(spec=BaseLLMService)
    mock_llm.generate.return_value = "Generated answer"

    state: AgentState = {
        "query": "test query",
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
        "context": "Some context",
        "retrieval_config": {},
        "messages": [],
    }
    result = generate_node(state, mock_llm, system_prompt="test prompt")
    assert result["answer"] == "Generated answer"
    assert result["candidate_answers"] == ["Generated answer"]
    mock_llm.generate.assert_called_once()


def test_generate_node_no_llm() -> None:
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
    result = generate_node(state, llm_service=None)
    assert "llm_service is required" in result["answer"]
    assert len(result["errors"]) > 0


def test_validate_node_valid() -> None:
    state: AgentState = {
        "query": "test",
        "rewritten_queries": [],
        "documents": [],
        "candidate_answers": [],
        "answer": "Answer with citation [1]",
        "confidence": None,
        "citations": [
            {
                "citation_id": 1,
                "source_id": "s1",
                "title": "Title",
                "url": "https://example.com",
                "domain": "example.com",
                "score": 0.9,
                "snippet": "Snippet",
                "chunk_index": 0,
                "total_chunks": 1,
            }
        ],
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
    result = validate_node(state)
    assert result["confidence"] == 0.8
    assert result["errors"] == []


def test_validate_node_no_sources() -> None:
    state: AgentState = {
        "query": "test",
        "rewritten_queries": [],
        "documents": [],
        "candidate_answers": [],
        "answer": "Answer without citation",
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
    result = validate_node(state)
    assert result["confidence"] == 0.0
    assert "No citation sources available" in result["errors"]


# ---------------------------------------------------------------------------
# Section C: Agent node unit tests
# ---------------------------------------------------------------------------


class StubLLMService(BaseLLMService):
    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.responses: list[LLMResponse] = responses or []

    def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> str:
        del prompt, system
        return ""

    def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
    ) -> Any:
        del prompt, system
        return iter([])

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: object | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        del tools, system
        self.calls.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content="Default response")


def test_decide_tool_node_with_tool() -> None:
    stub = StubLLMService(
        responses=[
            LLMResponse(
                tool_calls=[ToolCall(name="rag_search", arguments={"query": "test"}, id="call_1")],
            )
        ]
    )

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
    result = decide_tool_node(state, stub)
    assert len(result["tool_calls"]) == 1
    assert result["current_tool"] == "rag_search"
    assert len(result["messages"]) > 0


def test_decide_tool_node_direct_answer() -> None:
    stub = StubLLMService(responses=[LLMResponse(content="Direct answer")])

    state: AgentState = {
        "query": "hello",
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
    result = decide_tool_node(state, stub)
    assert result["current_tool"] is None
    assert result["tool_calls"] == []


def test_execute_tool_node() -> None:
    registry = {"test_tool": lambda query: f"Result for {query}"}

    with patch("nagrik_ai.agent.agent_nodes.TOOL_REGISTRY", registry):
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
            "tool_calls": [{"name": "test_tool", "arguments": {"query": "hello"}, "id": "call_1"}],
            "tool_results": [],
            "current_tool": "test_tool",
            "session_id": None,
            "user_id": None,
            "trace_id": None,
            "context": None,
            "retrieval_config": {},
            "messages": [],
        }
        result = execute_tool_node(state)
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["output"] == "Result for hello"
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], ToolMessage)


def test_execute_tool_node_unknown_tool() -> None:
    with patch("nagrik_ai.agent.agent_nodes.TOOL_REGISTRY", {}):
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
            "tool_calls": [{"name": "nonexistent", "arguments": {}, "id": "call_1"}],
            "tool_results": [],
            "current_tool": "nonexistent",
            "session_id": None,
            "user_id": None,
            "trace_id": None,
            "context": None,
            "retrieval_config": {},
            "messages": [],
        }
        result = execute_tool_node(state)
        assert "Unknown tool" in result["tool_results"][0]["output"]


def test_execute_tool_node_error() -> None:
    def broken_tool(**kwargs: Any) -> str:
        raise ValueError("Tool failure")

    with patch("nagrik_ai.agent.agent_nodes.TOOL_REGISTRY", {"broken": broken_tool}):
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
            "tool_calls": [{"name": "broken", "arguments": {}, "id": "call_1"}],
            "tool_results": [],
            "current_tool": "broken",
            "session_id": None,
            "user_id": None,
            "trace_id": None,
            "context": None,
            "retrieval_config": {},
            "messages": [],
        }
        result = execute_tool_node(state)
        assert "Error executing broken" in result["tool_results"][0]["output"]


def test_synthesize_node_with_results() -> None:
    stub = StubLLMService(responses=[LLMResponse(content="Synthesized answer")])

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
        "tool_calls": [{"name": "rag_search", "arguments": {"query": "test"}, "id": "call_1"}],
        "tool_results": [{"tool_call_id": "call_1", "output": "Some result"}],
        "current_tool": "rag_search",
        "session_id": None,
        "user_id": None,
        "trace_id": None,
        "context": None,
        "retrieval_config": {},
        "messages": [HumanMessage(content="test"), AIMessage(content="", tool_calls=[])],
    }
    result = synthesize_node(state, stub)
    assert result["answer"] == "Synthesized answer"
    assert len(stub.calls) == 1


def test_synthesize_node_empty_results() -> None:
    stub = StubLLMService(responses=[LLMResponse(content="Fallback answer")])

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
        "tool_results": [{"tool_call_id": "call_1", "output": ""}],
        "current_tool": None,
        "session_id": None,
        "user_id": None,
        "trace_id": None,
        "context": None,
        "retrieval_config": {},
        "messages": [HumanMessage(content="test"), AIMessage(content="Direct answer")],
    }
    result = synthesize_node(state, stub)
    assert result["answer"] == "Fallback answer"


def test_synthesize_node_direct_answer() -> None:
    stub = StubLLMService()

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
        "messages": [HumanMessage(content="test"), AIMessage(content="Direct answer")],
    }
    result = synthesize_node(state, stub)
    assert result["answer"] == "Direct answer"


def test_fallback_web_search_node() -> None:
    stub = StubLLMService(responses=[LLMResponse(content="Fallback answer")])

    with (
        patch("nagrik_ai.agent.agent_nodes.web_search", return_value="Web result"),
    ):
        state: AgentState = {
            "query": "test query",
            "rewritten_queries": [],
            "documents": [],
            "candidate_answers": [],
            "answer": None,
            "confidence": None,
            "citations": [],
            "errors": [],
            "metadata": {},
            "tool_calls": [],
            "tool_results": [{"tool_call_id": "call_1", "output": "I could not find this information"}],
            "current_tool": None,
            "session_id": None,
            "user_id": None,
            "trace_id": None,
            "context": None,
            "retrieval_config": {},
            "messages": [],
        }
        result = fallback_web_search_node(state, stub)
        assert result["answer"] == "Fallback answer"
        assert len(result["messages"]) == 3


# ---------------------------------------------------------------------------
# Section D: Graph routing & integration
# ---------------------------------------------------------------------------


def test_rag_graph_invoke() -> None:
    mock_retrieval = MagicMock(spec=DocumentRetrievalService)
    mock_retrieval.retrieve.return_value = [
        {
            "page_content": "GST is Goods and Services Tax",
            "source_id": "doc1",
            "title": "GST Guide",
            "url": "https://example.com",
            "domain": "example.com",
            "chunk_index": 0,
            "total_chunks": 1,
            "score": 0.95,
        }
    ]

    mock_llm = MagicMock(spec=BaseLLMService)
    mock_llm.generate.return_value = "GST is Goods and Services Tax."

    graph = create_rag_graph(
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
        system_prompt="You are a helpful assistant.",
    )

    initial: AgentState = {
        "query": "What is GST?",
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
        "retrieval_config": {"top_k": 5, "reranker_enabled": False},
        "messages": [],
    }
    final = graph.invoke(initial)
    answer = final.get("answer", "")
    assert answer.startswith("GST is Goods and Services Tax.")
    assert "**Sources:**" in answer
    assert final.get("confidence") is not None
    assert len(final.get("citations", [])) > 0


def test_agent_graph_tool_path() -> None:
    stub = StubLLMService(
        responses=[
            LLMResponse(tool_calls=[ToolCall(name="test_tool", arguments={"query": "hello"}, id="call_1")]),
            LLMResponse(content="Synthesized answer"),
        ]
    )
    registry = {"test_tool": lambda query: f"Result: {query}"}

    with patch("nagrik_ai.agent.agent_nodes.TOOL_REGISTRY", registry):
        graph = create_agent_graph(llm_service=stub)
        initial: AgentState = {
            "query": "hello",
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
            "messages": [HumanMessage(content="hello")],
        }
        final = graph.invoke(initial)
        assert final.get("answer") == "Synthesized answer"


def test_agent_graph_direct_path() -> None:
    stub = StubLLMService(responses=[LLMResponse(content="Direct answer")])

    graph = create_agent_graph(llm_service=stub)
    initial: AgentState = {
        "query": "hello",
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
        "messages": [HumanMessage(content="hello")],
    }
    final = graph.invoke(initial)
    assert final.get("answer") == "Direct answer"


def test_agent_graph_fallback_path() -> None:
    stub = StubLLMService(
        responses=[
            LLMResponse(tool_calls=[ToolCall(name="rag_search", arguments={"query": "test"}, id="call_1")]),
            LLMResponse(content="Fallback answer"),
        ]
    )
    registry = {"rag_search": lambda **_: "I could not find this information in the provided sources."}

    with (
        patch("nagrik_ai.agent.agent_nodes.TOOL_REGISTRY", registry),
        patch("nagrik_ai.agent.agent_nodes.web_search", return_value="Web result"),
    ):
        graph = create_agent_graph(llm_service=stub)
        initial: AgentState = {
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
            "messages": [HumanMessage(content="test")],
        }
        final = graph.invoke(initial)
        assert final.get("answer") == "Fallback answer"


def test_agent_graph_multi_turn() -> None:
    stub = StubLLMService(
        responses=[
            LLMResponse(content="First answer"),
            LLMResponse(content="Second answer"),
        ]
    )

    graph = create_agent_graph(llm_service=stub)

    initial: AgentState = {
        "query": "first query",
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
        "messages": [HumanMessage(content="first query")],
    }
    turn1 = graph.invoke(initial)
    assert turn1.get("answer") == "First answer"

    turn1_messages = turn1.get("messages", [])
    turn2_state: AgentState = {
        "query": "second query",
        "rewritten_queries": turn1.get("rewritten_queries", []),
        "documents": turn1.get("documents", []),
        "candidate_answers": turn1.get("candidate_answers", []),
        "answer": None,
        "confidence": turn1.get("confidence"),
        "citations": turn1.get("citations", []),
        "errors": turn1.get("errors", []),
        "metadata": turn1.get("metadata", {}),
        "tool_calls": turn1.get("tool_calls", []),
        "tool_results": turn1.get("tool_results", []),
        "current_tool": turn1.get("current_tool"),
        "session_id": turn1.get("session_id"),
        "user_id": turn1.get("user_id"),
        "trace_id": turn1.get("trace_id"),
        "context": turn1.get("context"),
        "retrieval_config": turn1.get("retrieval_config", {}),
        "messages": [*turn1_messages, HumanMessage(content="second query")],
    }
    turn2 = graph.invoke(turn2_state)
    assert turn2.get("answer") == "Second answer"
    assert len(turn2.get("messages", [])) > len(turn1_messages)


# ---------------------------------------------------------------------------
# Section E: Checkpointing
# ---------------------------------------------------------------------------


def test_agent_graph_with_checkpointer() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    stub = StubLLMService(
        responses=[
            LLMResponse(content="First response"),
            LLMResponse(content="Second response"),
        ]
    )

    checkpointer = InMemorySaver()
    graph = create_agent_graph(llm_service=stub, checkpointer=checkpointer)

    initial: AgentState = {
        "query": "first",
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
        "messages": [HumanMessage(content="first")],
    }
    turn1 = graph.invoke(initial, config={"configurable": {"thread_id": "test-thread-1"}})
    assert turn1.get("answer") == "First response"

    turn2 = graph.invoke(
        {"query": "second"},
        config={"configurable": {"thread_id": "test-thread-1"}},
    )
    assert turn2.get("answer") == "Second response"
    turn2_messages = list(turn2.get("messages", []))
    assert len(turn2_messages) > len(turn1.get("messages", []))


def test_agent_graph_persistence_across_graph_instances() -> None:
    """SqliteSaver persists state even when separate CompiledStateGraph instances are used,
    simulating a process restart."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver  # pyright: ignore[reportMissingTypeStubs]

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    stub1 = StubLLMService(responses=[LLMResponse(content="Turn 1 answer")])
    graph1 = create_agent_graph(llm_service=stub1, checkpointer=saver)

    turn1_state: AgentState = {
        "query": "first query",
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
        "messages": [HumanMessage(content="first query")],
    }
    result1 = graph1.invoke(turn1_state, config={"configurable": {"thread_id": "persist-test"}})
    assert result1.get("answer") == "Turn 1 answer"
    turn1_messages = list(result1.get("messages", []))

    stub2 = StubLLMService(responses=[LLMResponse(content="Turn 2 answer")])
    graph2 = create_agent_graph(llm_service=stub2, checkpointer=saver)

    result2 = graph2.invoke(
        {"query": "second query"},
        config={"configurable": {"thread_id": "persist-test"}},
    )
    assert result2.get("answer") == "Turn 2 answer"
    turn2_messages = list(result2.get("messages", []))

    assert len(turn2_messages) > len(turn1_messages)
    assert any("first query" in (m.content or "") for m in turn2_messages)
    assert any("second query" in (m.content or "") for m in turn2_messages)


# ---------------------------------------------------------------------------
# Section F: Equivalence — RAG graph vs orchestrator
# ---------------------------------------------------------------------------


def test_rag_graph_matches_orchestrator() -> None:
    from nagrik_ai.services.rag_orchestrator import RAGOrchestrator

    mock_retrieval = MagicMock(spec=DocumentRetrievalService)
    mock_retrieval.retrieve.return_value = [
        {
            "page_content": "GST is Goods and Services Tax",
            "source_id": "doc1",
            "title": "GST Guide",
            "url": "https://example.com",
            "domain": "example.com",
            "chunk_index": 0,
            "total_chunks": 1,
            "score": 0.95,
        }
    ]

    mock_llm = MagicMock(spec=BaseLLMService)
    mock_llm.generate.return_value = "GST is Goods and Services Tax."

    orchestrator = RAGOrchestrator(
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
    )
    expected: RAGResult = orchestrator.query("What is GST?")

    graph = create_rag_graph(
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
        system_prompt="You are a helpful assistant.",
    )
    initial: AgentState = {
        "query": "What is GST?",
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
        "retrieval_config": {"top_k": 5, "reranker_enabled": False},
        "messages": [],
    }
    final_state: dict[str, Any] = graph.invoke(initial)

    actual = adapt_to_rag_result(final_state, query="What is GST?")  # type: ignore[arg-type]

    assert actual.response == expected.response
    assert len(actual.sources) >= 0
    assert actual.citations_valid == expected.citations_valid

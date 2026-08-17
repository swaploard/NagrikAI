from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from nagrik_ai.agent.agent_graph import create_agent_graph
from nagrik_ai.agent.agent_nodes import (
    decide_tool_node,
    execute_tool_node,
    fallback_web_search_node,
    synthesize_node,
)
from nagrik_ai.agent.nodes import (
    build_context_node,
    classify_node,
    generate_node,
    rerank_node,
    response_token_budget,
    retrieve_node,
    validate_node,
)
from nagrik_ai.agent.rag_graph import build_initial_state, create_rag_graph
from nagrik_ai.config.config_models import (
    MAX_RESPONSE_TOKENS,
    MAX_RESPONSE_TOKENS_DETAILED,
    MAX_RESPONSE_TOKENS_HARD,
)
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
    assert "classify" in graph.nodes
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
    assert "classify" in mermaid
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


def test_classify_node() -> None:
    state: AgentState = {
        "query": "How to file GSTR-1?",
        "rewritten_queries": [],
        "documents": [],
        "candidate_answers": [],
        "answer": None,
        "confidence": None,
        "citations": [],
        "errors": [],
        "metadata": {"existing": "kept"},
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
    result = classify_node(state)
    metadata = result["metadata"]
    assert metadata["existing"] == "kept"
    assert metadata["verbosity"] == "detailed"
    assert metadata["question_type"] == "PROCEDURAL"


def test_classify_node_defaults() -> None:
    state: AgentState = {
        "query": "What is the due date for GSTR-3B?",
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
    result = classify_node(state)
    metadata = result["metadata"]
    assert metadata["verbosity"] == "concise"
    assert metadata["question_type"] == "FACTUAL"


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


def test_build_context_rank_citations_by_authority() -> None:
    """Higher-authority sources (Rule) must get a lower citation_id than a FAQ."""
    state: AgentState = {
        "query": "test query",
        "rewritten_queries": [],
        "documents": [
            {
                "content": "GST FAQ answer text.",
                "metadata": {
                    "source_id": "faq_1",
                    "title": "FAQs_GST",
                    "url": "https://gst.gov.in/faq",
                    "domain": "gst.gov.in",
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
                "score": 0.95,
            },
            {
                "content": "Relevant GST Rule text.",
                "metadata": {
                    "source_id": "rule_1",
                    "title": "GST Rules",
                    "url": "https://gst.gov.in/rules",
                    "domain": "gst.gov.in",
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
                "score": 0.90,
            },
        ],
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
    citations = result["citations"]
    assert [c["citation_id"] for c in citations] == [1, 2]
    assert citations[0]["source_id"] == "rule_1"
    assert citations[1]["source_id"] == "faq_1"
    context = result["context"]
    assert context.index("[1]") < context.index("[2]")


def test_build_context_citation_order_is_deterministic() -> None:
    """Same authority + same score must tie-break lexicographically (URL, then title)."""

    def _doc(source_id: str, url: str, title: str) -> dict[str, Any]:
        return {
            "content": f"{source_id} content.",
            "metadata": {
                "source_id": source_id,
                "title": title,
                "url": url,
                "domain": "gst.gov.in",
                "chunk_index": 0,
                "total_chunks": 1,
            },
            "score": 0.90,
        }

    state: AgentState = {
        "query": "test query",
        "rewritten_queries": [],
        "documents": [
            _doc("rule_b", "https://gst.gov.in/rules-z", "GST Rules Z"),
            _doc("rule_a", "https://gst.gov.in/rules-a", "GST Rules A"),
        ],
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
    citations = result["citations"]
    assert [c["citation_id"] for c in citations] == [1, 2]
    assert citations[0]["source_id"] == "rule_a"
    assert citations[1]["source_id"] == "rule_b"


def test_build_context_merges_chunks_per_source() -> None:
    """All chunks of the same source must appear under ONE merged context block."""
    state: AgentState = {
        "query": "gst registration threshold",
        "rewritten_queries": [],
        "documents": [
            {
                "content": "chunk zero content.",
                "metadata": {
                    "source_id": "guide_1",
                    "source_type": "user_guide",
                    "title": "GST Registration Guide",
                    "url": "https://gst.gov.in/registration-guide",
                    "domain": "gst.gov.in",
                    "chunk_index": 0,
                    "total_chunks": 2,
                },
                "score": 0.9,
            },
            {
                "content": "chunk one content.",
                "metadata": {
                    "source_id": "guide_1",
                    "source_type": "user_guide",
                    "title": "GST Registration Guide",
                    "url": "https://gst.gov.in/registration-guide",
                    "domain": "gst.gov.in",
                    "chunk_index": 1,
                    "total_chunks": 2,
                },
                "score": 0.8,
            },
            {
                "content": "faq content.",
                "metadata": {
                    "source_id": "faq_1",
                    "source_type": "faq",
                    "title": "GST FAQ",
                    "url": "https://gst.gov.in/faq/registration",
                    "domain": "gst.gov.in",
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
                "score": 0.95,
            },
        ],
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
    citations = result["citations"]
    assert len(citations) == 2, "Two distinct sources, even with multiple chunks"
    assert {c["source_id"] for c in citations} == {"guide_1", "faq_1"}
    assert [c["citation_id"] for c in citations] == [1, 2]

    context = result["context"]
    assert context.count("[1]") == 1, "Source must appear once per merged block"
    assert context.count("[2]") == 1
    assert "Chunk 0:\nchunk zero content." in context
    assert "Chunk 1:\nchunk one content." in context
    assert "Chunk 0:\nfaq content." in context


def test_build_context_merged_block_preserves_lock() -> None:
    """Citation IDs set in the merged context must match the returned citations list."""
    state: AgentState = {
        "query": "test query",
        "rewritten_queries": [],
        "documents": [
            {
                "content": "first chunk.",
                "metadata": {
                    "source_id": "doc_a",
                    "title": "Doc A",
                    "url": "https://example.com/doc-a",
                    "domain": "example.com",
                    "chunk_index": 0,
                    "total_chunks": 2,
                },
                "score": 0.9,
            },
            {
                "content": "second chunk.",
                "metadata": {
                    "source_id": "doc_a",
                    "title": "Doc A",
                    "url": "https://example.com/doc-a",
                    "domain": "example.com",
                    "chunk_index": 1,
                    "total_chunks": 2,
                },
                "score": 0.85,
            },
        ],
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
    assert len(result["citations"]) == 1
    assert result["citations"][0]["citation_id"] == 1
    assert len(result["documents"]) == 2
    assert all(doc["citation_id"] == 1 for doc in result["documents"])


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


def _long_answer(word_count: int) -> str:
    return " ".join(["word"] * word_count) + " [1]"


def test_validate_node_length_guard_concise_overshoot() -> None:
    state: AgentState = {
        "query": "What is the due date for GSTR-3B?",
        "rewritten_queries": [],
        "documents": [],
        "candidate_answers": [],
        "answer": _long_answer(600),
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
        "metadata": {"verbosity": "concise"},
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
    assert result["confidence"] == 0.5
    assert result["errors"] == []
    assert result["metadata"]["length_warning"] is True
    assert result["metadata"]["response_word_count"] == 601
    assert result["citations_valid"] is True


def test_validate_node_length_guard_concise_under_cap() -> None:
    state: AgentState = {
        "query": "What is the due date for GSTR-3B?",
        "rewritten_queries": [],
        "documents": [],
        "candidate_answers": [],
        "answer": _long_answer(100),
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
        "metadata": {"verbosity": "concise"},
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
    assert "length_warning" not in result["metadata"]
    assert result["metadata"]["response_word_count"] == 101


def test_validate_node_length_guard_detailed_not_penalized() -> None:
    state: AgentState = {
        "query": "How to file GSTR-1?",
        "rewritten_queries": [],
        "documents": [],
        "candidate_answers": [],
        "answer": _long_answer(900),
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
        "metadata": {"verbosity": "detailed"},
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
    assert "length_warning" not in result["metadata"]


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


def test_rag_graph_injects_classification_into_prompt() -> None:
    """The classify node must run first and feed verbosity/question_type into generation."""
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
    mock_llm.generate.return_value = "File GSTR-1 monthly. [1]"

    graph = create_rag_graph(
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
        system_prompt="You are a helpful assistant.",
    )

    initial: AgentState = {
        "query": "How to file GSTR-1?",
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
    metadata = final.get("metadata", {})
    assert metadata.get("verbosity") == "detailed"
    assert metadata.get("question_type") == "PROCEDURAL"

    prompt_arg = mock_llm.generate.call_args.args[0]
    assert "VERBOSITY DIRECTIVE: detailed" in prompt_arg
    assert "QUESTION TYPE: PROCEDURAL" in prompt_arg
    assert "How to file GSTR-1?" in prompt_arg


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
# Section F: Conditional flows — fallback & self-correction
# ---------------------------------------------------------------------------


def test_rag_graph_compiles_with_fallback() -> None:
    graph = create_rag_graph(
        retrieval_service=MagicMock(spec=DocumentRetrievalService),
        llm_service=MagicMock(spec=BaseLLMService),
        system_prompt="test",
        enable_fallback=True,
    )
    assert isinstance(graph, CompiledStateGraph)
    assert "web_search_fallback" in graph.nodes


def test_rag_graph_compiles_with_self_correction() -> None:
    graph = create_rag_graph(
        retrieval_service=MagicMock(spec=DocumentRetrievalService),
        llm_service=MagicMock(spec=BaseLLMService),
        system_prompt="test",
        enable_self_correction=True,
    )
    assert isinstance(graph, CompiledStateGraph)
    assert "retry_generate" in graph.nodes


def test_rag_graph_fallback_path() -> None:
    """When citations are invalid and enable_fallback is True, graph routes to web_search_fallback."""
    mock_retrieval = MagicMock(spec=DocumentRetrievalService)
    mock_retrieval.retrieve.return_value = []

    mock_llm = MagicMock(spec=BaseLLMService)
    mock_llm.generate.return_value = "No information found in sources."

    with patch("nagrik_ai.agent.rag_graph.web_search", return_value="Web result") as mock_web:
        graph = create_rag_graph(
            retrieval_service=mock_retrieval,
            llm_service=mock_llm,
            system_prompt="You are a helpful assistant.",
            enable_fallback=True,
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
        mock_web.assert_called_once()
        answer = final.get("answer", "")
        assert len(answer) > 0
        assert "rag_result" in final


def test_rag_graph_self_correction_path() -> None:
    """When citations are invalid and enable_self_correction is True, graph retries generation."""
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
            "metadata": {
                "source_id": "doc1",
                "title": "GST Guide",
                "url": "https://example.com",
                "domain": "example.com",
            },
        }
    ]

    mock_llm = MagicMock(spec=BaseLLMService)
    mock_llm.generate.return_value = "Answer with invalid citation [99]"

    graph = create_rag_graph(
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
        system_prompt="You are a helpful assistant.",
        enable_self_correction=True,
        max_retries=2,
    )

    initial: AgentState = {
        "query": "What is GST?",
        "rewritten_queries": [],
        "documents": [],
        "candidate_answers": [],
        "answer": None,
        "confidence": None,
        "citations": [
            {
                "citation_id": 1,
                "source_id": "doc1",
                "title": "GST Guide",
                "url": "https://example.com",
                "domain": "example.com",
                "score": 0.95,
                "snippet": "Content",
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
        "retrieval_config": {"top_k": 5, "reranker_enabled": False},
        "messages": [],
    }
    final = graph.invoke(initial)
    retry_count = final.get("metadata", {}).get("retry_count", 0)
    assert retry_count > 0, "Self-correction should have incremented retry_count"


def test_rag_graph_self_correction_exhausts_retries() -> None:
    """When max_retries is reached, self-correction stops retrying and moves to finalize."""
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
            "metadata": {
                "source_id": "doc1",
                "title": "GST Guide",
                "url": "https://example.com",
                "domain": "example.com",
            },
        }
    ]

    mock_llm = MagicMock(spec=BaseLLMService)
    mock_llm.generate.return_value = "Answer with invalid citation [99]"

    graph = create_rag_graph(
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
        system_prompt="You are a helpful assistant.",
        enable_self_correction=True,
        max_retries=1,
    )

    initial: AgentState = {
        "query": "What is GST?",
        "rewritten_queries": [],
        "documents": [],
        "candidate_answers": [],
        "answer": None,
        "confidence": None,
        "citations": [
            {
                "citation_id": 1,
                "source_id": "doc1",
                "title": "GST Guide",
                "url": "https://example.com",
                "domain": "example.com",
                "score": 0.95,
                "snippet": "Content",
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
        "retrieval_config": {"top_k": 5, "reranker_enabled": False},
        "messages": [],
    }
    final = graph.invoke(initial)
    retry_count = final.get("metadata", {}).get("retry_count", 0)
    assert retry_count == 1, f"Expected 1 retry with max_retries=1, got {retry_count}"
    assert "rag_result" in final


def test_rag_graph_self_correction_success_on_retry() -> None:
    """When retry generates valid citations, self-correction stops and proceeds to finalize."""
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
            "metadata": {
                "source_id": "doc1",
                "title": "GST Guide",
                "url": "https://example.com",
                "domain": "example.com",
            },
        }
    ]

    mock_llm = MagicMock(spec=BaseLLMService)
    mock_llm.generate.return_value = "GST is Goods and Services Tax. [1]"

    graph = create_rag_graph(
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
        system_prompt="You are a helpful assistant.",
        enable_self_correction=True,
        max_retries=2,
    )

    initial: AgentState = {
        "query": "What is GST?",
        "rewritten_queries": [],
        "documents": [],
        "candidate_answers": [],
        "answer": None,
        "confidence": None,
        "citations": [
            {
                "citation_id": 1,
                "source_id": "doc1",
                "title": "GST Guide",
                "url": "https://example.com",
                "domain": "example.com",
                "score": 0.95,
                "snippet": "Content",
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
        "retrieval_config": {"top_k": 5, "reranker_enabled": False},
        "messages": [],
    }
    final = graph.invoke(initial)
    result = final.get("rag_result")
    assert result is not None
    assert result.citations_valid is True
    assert "GST" in result.response


# ---------------------------------------------------------------------------
# Section G: Equivalence — RAG graph vs orchestrator
# ---------------------------------------------------------------------------


def test_rag_graph_produces_rag_result() -> None:
    """Verify the RAG graph produces a RAGResult with the expected fields."""
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
            "metadata": {
                "source_id": "doc1",
                "title": "GST Guide",
                "url": "https://example.com",
                "domain": "example.com",
            },
        }
    ]

    mock_llm = MagicMock(spec=BaseLLMService)
    mock_llm.generate.return_value = "GST is Goods and Services Tax. [1]"

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
        "rag_result": None,
        "_streaming_buffer": None,
        "_streaming_callback": None,
    }
    final_state: dict[str, Any] = graph.invoke(initial)

    result = final_state.get("rag_result")
    assert result is not None
    assert isinstance(result, RAGResult)
    assert result.response.startswith("GST is Goods and Services Tax.")
    assert result.citations_valid is True


# ---------------------------------------------------------------------------
# Section H: Verbosity-aware token budget + truncation-aware retry
# ---------------------------------------------------------------------------


class _TruncationAwareLLM:
    """Fake LLM that flips ``last_finish_reason`` across calls.

    The first ``generate``/``generate_stream`` reports ``length`` (truncated), the
    remaining calls report ``stop``. The initial response is deliberately cut off.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.stream_buffers: list[list[str]] = []
        self.last_finish_reason: str | None = None

    def _bump(self) -> None:
        self.calls += 1
        self.last_finish_reason = "length" if self.calls == 1 else "stop"

    def generate(self, prompt: str, system: str | None = None, max_tokens: int | None = None) -> str:
        self._last_prompt = prompt
        self._last_system = system
        self._last_budget = max_tokens
        self._bump()
        if self.calls == 1:
            return "The filing deadline for each month"
        return "The filing deadline for each month is the 13th. [1]"

    def generate_stream(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None
    ) -> Any:
        self._last_prompt = prompt
        self._last_system = system
        self._last_budget = max_tokens
        tokens = ["The filing deadline ", "is the 13th. [1]"]
        if self.calls == 0:
            tokens = ["The filing dead"]
        self._bump()
        yield tokens[-1]

    def chat(self, *_args: Any, **_kwargs: Any) -> Any:
        return LLMResponse(content="Mong")


def _retrieval_doc() -> list[dict[str, Any]]:
    return [
        {
            "page_content": "IFF is filed quarterly. The due date is the 13th.",
            "source_id": "doc1",
            "title": "GST Guide",
            "url": "https://example.com",
            "domain": "example.com",
            "chunk_index": 0,
            "total_chunks": 1,
            "score": 0.95,
            "metadata": {
                "source_id": "doc1",
                "title": "GST Guide",
                "url": "https://example.com",
                "domain": "example.com",
            },
        }
    ]


def test_response_token_budget_scales_with_verbosity_and_retries() -> None:
    assert response_token_budget({}) == MAX_RESPONSE_TOKENS
    assert response_token_budget({"verbosity": "concise"}) == MAX_RESPONSE_TOKENS
    assert response_token_budget({"verbosity": "detailed"}) == MAX_RESPONSE_TOKENS_DETAILED
    assert response_token_budget({"verbosity": "concise", "retry_count": 1}) == min(
        2 * MAX_RESPONSE_TOKENS, MAX_RESPONSE_TOKENS_HARD
    )
    assert response_token_budget({"verbosity": "detailed", "retry_count": 2}) == min(
        4 * MAX_RESPONSE_TOKENS_DETAILED, MAX_RESPONSE_TOKENS_HARD
    )


def test_generate_node_passes_verbosity_aware_max_tokens() -> None:
    mock_llm = MagicMock(spec=BaseLLMService)
    mock_llm.generate.return_value = "Answer [1]"
    state = build_initial_state(
        query="What is GST?", retrieval_config={"top_k": 5, "reranker_enabled": False}
    )
    state["metadata"] = {"verbosity": "detailed", "question_type": "FACTUAL"}

    generate_node(state, mock_llm, system_prompt="You are a tax assistant.")
    _, kwargs = mock_llm.generate.call_args
    assert kwargs["max_tokens"] == MAX_RESPONSE_TOKENS_DETAILED


def test_rag_graph_truncation_retries_in_fallback_mode() -> None:
    """A truncated (finish_reason=length) answer triggers a retry even without self-correction."""
    mock_retrieval = MagicMock(spec=DocumentRetrievalService)
    mock_retrieval.retrieve.return_value = _retrieval_doc()
    mock_llm = _TruncationAwareLLM()

    graph = create_rag_graph(
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
        system_prompt="You are a helpful assistant.",
        enable_fallback=True,
        max_retries=2,
    )

    initial = build_initial_state(
        query="What is the IFF filing deadline?",
        retrieval_config={"top_k": 5, "reranker_enabled": False},
    )
    final = graph.invoke(initial)
    retry_count = final.get("metadata", {}).get("retry_count", 0)
    assert retry_count >= 1, "Truncated response should trigger a retry"
    assert final.get("metadata", {}).get("truncated", False) is False
    result = final.get("rag_result")
    assert result is not None
    assert "the 13th" in result.response


def test_rag_graph_truncation_exhausts_retries_then_finalizes() -> None:
    """When every attempt is truncated, the graph stops retrying and still finalizes."""
    mock_retrieval = MagicMock(spec=DocumentRetrievalService)
    mock_retrieval.retrieve.return_value = _retrieval_doc()

    class _AlwaysTruncated:
        def __init__(self) -> None:
            self.last_finish_reason: str | None = "length"

        def generate(self, prompt: str, system: str | None = None, max_tokens: int | None = None) -> str:
            self._last_prompt = prompt
            self._last_system = system
            self._last_budget = max_tokens
            return "Partial answer without a trailing citation."

        def generate_stream(
            self, prompt: str, system: str | None = None, max_tokens: int | None = None
        ) -> Iterator[str]:
            self._last_prompt = prompt
            self._last_system = system
            self._last_budget = max_tokens
            yield "Partial answer without a trailing citation."

        def chat(self, *_args: Any, **_kwargs: Any) -> Any:
            return LLMResponse(content="Mong")

    graph = create_rag_graph(
        retrieval_service=mock_retrieval,
        llm_service=_AlwaysTruncated(),
        system_prompt="You are a helpful assistant.",
        enable_fallback=True,
        max_retries=2,
    )

    initial = build_initial_state(
        query="What is the IFF filing deadline?",
        retrieval_config={"top_k": 5, "reranker_enabled": False},
    )
    final = graph.invoke(initial)
    assert final.get("metadata", {}).get("retry_count", 0) == 2
    assert final.get("rag_result") is not None


def test_streaming_retry_resets_stream_buffer() -> None:
    """The streaming buffer only carries the retry's tokens, not the truncated first attempt."""
    mock_retrieval = MagicMock(spec=DocumentRetrievalService)
    mock_retrieval.retrieve.return_value = _retrieval_doc()
    mock_llm = _TruncationAwareLLM()

    graph = create_rag_graph(
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
        system_prompt="You are a helpful assistant.",
        enable_streaming=True,
        enable_fallback=True,
        max_retries=2,
    )

    initial = build_initial_state(
        query="What is the IFF filing deadline?",
        retrieval_config={"top_k": 5, "reranker_enabled": False},
    )
    final = graph.invoke(initial)
    buffer = final.get("_streaming_buffer") or []
    assert "The filing dead" not in buffer
    assert any("is the 13th" in t for t in buffer)

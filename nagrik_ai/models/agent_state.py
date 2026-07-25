from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

from nagrik_ai.models.rag_result import RAGResult


class AgentState(TypedDict):
    # Core fields
    query: str
    rewritten_queries: list[str]
    documents: list[dict[str, Any]]
    candidate_answers: list[str]
    answer: str | None
    confidence: float | None
    citations: list[dict[str, Any]]
    errors: list[str]
    metadata: dict[str, Any]

    # Tool state
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    current_tool: str | None

    # Session context
    session_id: str | None
    user_id: str | None
    trace_id: str | None

    # Pipeline state
    context: str | None

    # Citation validation
    citations_valid: bool | None

    # Config
    retrieval_config: dict[str, Any]  # top_k, reranker_enabled, etc.

    # Conversation
    messages: list[BaseMessage]  # LangChain message objects

    # Tracing & result fields (Phase 1)
    rag_result: RAGResult | None  # final RAG result populated at end of pipeline
    _streaming_buffer: list[str] | None  # internal token accumulation buffer; not persisted
    _streaming_callback: Any  # callable injected at invoke time; type: ignore[typeddict-unknown-key]

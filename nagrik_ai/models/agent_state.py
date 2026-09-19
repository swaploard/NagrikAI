from typing import Any, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage

from nagrik_ai.models.rag_result import RAGResult
from nagrik_ai.models.tool_result import EvidencePolicy, ToolResult


class AgentStep(TypedDict):
    action: str
    action_input: dict[str, Any]
    observation_summary: str
    timestamp_ms: float | None
    latency_ms: float | None
    tool_result: ToolResult | None


class ClaimVerdict(TypedDict):
    claim: str
    evidence_ids: list[int]
    tool_result_refs: list[str]
    status: Literal["supported", "unsupported", "indeterminate"]
    method: Literal["deterministic", "citation_check", "authority_check", "llm_assisted"]


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

    # Agent orchestration state (Phase 0 contract)
    agent_steps: NotRequired[list[AgentStep]]
    claim_verdicts: NotRequired[list[ClaimVerdict]]
    evidence_policy: NotRequired[EvidencePolicy | None]
    business_context: NotRequired[dict[str, Any]]
    pending_profile_updates: NotRequired[dict[str, Any] | None]
    iteration: NotRequired[int]
    tool_calls_count: NotRequired[int]
    validation_retries: NotRequired[int]
    max_iterations: NotRequired[int]
    max_tool_calls: NotRequired[int]
    max_validation_retries: NotRequired[int]
    validation_errors: NotRequired[list[str]]
    missing_info: NotRequired[list[str]]
    finalized_with_limitations: NotRequired[bool]

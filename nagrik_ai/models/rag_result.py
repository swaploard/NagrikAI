"""RAG result dataclasses for structured responses with citations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SourceInfo:
    """Information about a source document for citation display."""

    title: str
    url: str
    domain: str
    source_id: str
    citation_id: int  # locked during context construction
    chunk_index: int
    total_chunks: int
    score: float = 0.0
    snippet: str = ""  # preview text for UI


@dataclass
class RAGResult:
    """Complete result from a RAG query with citations and metadata."""

    response: str
    sources: list[SourceInfo]  # deduplicated for display
    citation_map: dict[int, SourceInfo]  # original mapping for validation
    query: str  # observability / tracing
    raw_chunks: list[str]  # debugging / evaluation
    latency_ms: float  # performance tracking
    total_chunks_retrieved: int = 0
    citations_valid: bool = True  # populated by validate_citations()

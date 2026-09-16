from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SourceAuthority = Literal["authoritative", "secondary", "general_web", "deterministic"]
ErrorCode = Literal["TIMEOUT", "NOT_FOUND", "VALIDATION_ERROR", "RATE_LIMITED", "FORBIDDEN_TRANSITION", None]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: str | dict[str, Any] | list[Any] | None
    error_code: ErrorCode
    error_message: str | None
    source_authority: SourceAuthority | None
    latency_ms: float | None
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidencePolicy:
    """Per-question evidence requirement resolved before tool execution."""

    required_authority: SourceAuthority | None
    required_tools: frozenset[str] = frozenset()
    forbidden_tools: frozenset[str] = frozenset()
    description: str = ""

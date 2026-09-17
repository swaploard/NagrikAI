from __future__ import annotations

import re

from nagrik_ai.models.tool_result import EvidencePolicy, ToolResult


class ToolSelectionPolicy:
    """Runtime tool policy used by the orchestrator.

    Golden datasets verify this policy from the outside; they are not runtime inputs.
    """

    def validate_tool_call(self, tool: str, evidence_policy: EvidencePolicy | None) -> ToolResult | None:
        if evidence_policy is None:
            return None
        if tool in evidence_policy.forbidden_tools:
            return ToolResult(
                ok=False,
                data=None,
                error_code="VALIDATION_ERROR",
                error_message=f"Tool '{tool}' is forbidden for this evidence policy.",
                source_authority=None,
                latency_ms=None,
            )
        if evidence_policy.required_authority == "authoritative" and tool == "web_search":
            return ToolResult(
                ok=False,
                data=None,
                error_code="VALIDATION_ERROR",
                error_message="Authoritative source required before using general web evidence.",
                source_authority=None,
                latency_ms=None,
            )
        return None

    def resolve_evidence_policy(self, query: str, question_type: str | None = None) -> EvidencePolicy:
        text = f"{question_type or ''} {query}".lower()
        calculation = question_type in {"deterministic_calc", "calculation"} or bool(
            re.search(r"\b(calculate|calculator|gst at|emi|total)\b", text)
        )
        statutory = question_type in {"statutory_interpretation", "legal", "regulatory"} or bool(
            re.search(
                r"\b(act|rule|section|rate|threshold|eligibility|statutory|legal|regulatory|"
                r"composition|registration|compliance|itc|gstr|iff)\b",
                text,
            )
        )
        tools: set[str] = set()
        if calculation:
            tools.add("calculator")
        if statutory:
            tools.add("rag_search")
        return EvidencePolicy(
            required_authority="authoritative" if statutory else None,
            required_tools=frozenset(tools),
            forbidden_tools=frozenset({"web_search"}) if statutory else frozenset(),
            description="statutory_interpretation"
            if statutory
            else ("deterministic_calc" if calculation else "operational_info"),
        )

    def check_forbidden_transition(
        self,
        prev_step: str | None,
        next_tool: str,
        evidence_policy: EvidencePolicy | None,
    ) -> bool:
        return (
            prev_step == "web_search"
            and next_tool == "synthesize"
            and evidence_policy is not None
            and evidence_policy.required_authority == "authoritative"
        )

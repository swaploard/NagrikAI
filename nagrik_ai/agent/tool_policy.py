from __future__ import annotations

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
        if any(term in text for term in ("calculate", "calculator", "gst at", "interest", "penalty")):
            return EvidencePolicy(
                required_authority=None,
                required_tools=frozenset({"calculator"}),
                description="deterministic_calc",
            )
        if any(term in text for term in ("act", "rule", "section", "rate", "threshold", "eligib")):
            return EvidencePolicy(
                required_authority="authoritative",
                required_tools=frozenset({"rag_search"}),
                forbidden_tools=frozenset({"web_search"}),
                description="statutory_interpretation",
            )
        return EvidencePolicy(required_authority=None, description="operational_info")

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

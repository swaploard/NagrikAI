"""Conservative Phase 1 gates; semantic claim validation is added in Phase 3."""

from __future__ import annotations

import re
from typing import Any

from nagrik_ai.agent.react_nodes import step
from nagrik_ai.agent.tool_policy import ToolSelectionPolicy
from nagrik_ai.config.config_models import MAX_VALIDATION_RETRIES
from nagrik_ai.models.agent_state import AgentState, ClaimVerdict


def validate_evidence_node(state: AgentState, tool_policy: ToolSelectionPolicy | None = None) -> dict[str, Any]:
    policy = tool_policy or ToolSelectionPolicy()
    evidence = state.get("evidence_policy")
    successful = [s for s in state.get("agent_steps", []) if s["tool_result"] and s["tool_result"].ok]
    errors: list[str] = []
    if evidence:
        missing = evidence.required_tools - {s["action"] for s in successful}
        if missing:
            errors.append("Missing successful tools: " + ", ".join(sorted(missing)))
        if evidence.required_authority and not any(
            result.source_authority == evidence.required_authority
            for s in successful
            if (result := s["tool_result"]) is not None
        ):
            errors.append(f"Missing {evidence.required_authority} evidence.")
        if "calculator" in evidence.required_tools and not any(
            result.source_authority == "deterministic" for s in successful if (result := s["tool_result"]) is not None
        ):
            errors.append("Missing deterministic calculation evidence.")
    if state.get("tool_results") and not successful:
        errors.append("No successful tool evidence is available.")
    previous = next((s["action"] for s in reversed(state.get("agent_steps", [])) if s["tool_result"]), None)
    if policy.check_forbidden_transition(previous, "synthesize", evidence):
        errors.append("FORBIDDEN_TRANSITION: cannot synthesize after this evidence source.")
    return {
        "validation_errors": errors,
        "validation_retries": min(
            state.get("validation_retries", 0) + bool(errors),
            state.get("max_validation_retries", MAX_VALIDATION_RETRIES),
        ),
        "agent_steps": [
            *state.get("agent_steps", []),
            step("validate_evidence", "; ".join(errors) or "Evidence requirements satisfied."),
        ],
    }


def validate_claims_node(state: AgentState) -> dict[str, Any]:
    """Check resolvable citations; never label unverified tool-backed prose as supported."""
    answer = state.get("answer") or ""
    successful = [
        s["tool_result"] for s in state.get("agent_steps", []) if s["tool_result"] is not None and s["tool_result"].ok
    ]
    source_ids = {sid for result in successful for sid in result.source_ids}
    citation_ids: set[str] = set()
    for result in successful:
        if isinstance(result.data, dict):
            for source in result.data.get("sources", []):
                if str(source.get("source_id", "")) in source_ids:
                    citation_ids.add(str(source.get("citation_id")))
    verdicts: list[ClaimVerdict] = []
    for citation in dict.fromkeys(re.findall(r"\[(\d+)\]", answer)):
        verdicts.append(
            {
                "claim": f"Citation [{citation}] resolves to a retrieved source.",
                "evidence_ids": [int(citation)],
                "tool_result_refs": [],
                "status": "supported" if citation in citation_ids else "unsupported",
                "method": "citation_check",
            }
        )
    if successful:
        # Exact deterministic output can be checked without a semantic judge.
        exact = next(
            (r for r in successful if r.source_authority == "deterministic" and str(r.data) == answer.strip()), None
        )
        verdicts.append(
            {
                "claim": answer,
                "evidence_ids": [],
                "tool_result_refs": list(exact.source_ids) if exact else sorted(source_ids),
                "status": "supported" if exact else "indeterminate",
                "method": "deterministic" if exact else "citation_check",
            }
        )
    elif not answer.strip():
        verdicts.append(
            {
                "claim": "An answer is available.",
                "evidence_ids": [],
                "tool_result_refs": [],
                "status": "unsupported",
                "method": "citation_check",
            }
        )
    unresolved = any(v["status"] != "supported" for v in verdicts)
    errors = ["Answer claims require further validation."] if unresolved else []
    updates: dict[str, Any] = {
        "claim_verdicts": verdicts,
        "validation_errors": errors,
        "validation_retries": min(
            state.get("validation_retries", 0) + unresolved, state.get("max_validation_retries", MAX_VALIDATION_RETRIES)
        ),
        "metadata": {**state.get("metadata", {}), "agent_turn_completed": not unresolved},
        "agent_steps": [
            *state.get("agent_steps", []),
            step("validate_claims", "; ".join(errors) or "Available claim checks passed."),
        ],
    }
    if any(v["status"] == "unsupported" for v in verdicts):
        updates.update(citations_valid=False, confidence=0.4)
    elif unresolved:
        updates["confidence"] = 0.4
    return updates

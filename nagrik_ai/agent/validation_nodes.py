"""Evidence gates and deterministic-first claim validation for the bounded agent."""

from __future__ import annotations

import json
import re
from copy import copy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from nagrik_ai.agent.react_nodes import step
from nagrik_ai.agent.tool_policy import ToolSelectionPolicy
from nagrik_ai.agent.validation import validation_body
from nagrik_ai.config.config_models import MAX_VALIDATION_RETRIES
from nagrik_ai.models.agent_state import AgentState, ClaimVerdict
from nagrik_ai.models.rag_result import SourceInfo
from nagrik_ai.models.tool_result import ToolResult
from nagrik_ai.prompts.prompt_loader import load_prompt
from nagrik_ai.services.citation_service import validate_citations
from nagrik_ai.services.llm_service import BaseLLMService
from nagrik_ai.services.tracing import LangSmithTracer, get_tracer

_CITATION = re.compile(r"\[(\d+)\]")
_LEGAL = re.compile(r"\b(section|rule|act|statutory|legal|regulatory|must|required|eligible|eligibility)\b", re.I)
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?!\w|\.\d)")
_LEGAL_NUMBER = re.compile(r"\b(?:section|rule|article)\s+\d+(?:\([^)]*\))*", re.I)


@dataclass(frozen=True)
class EvidenceIndex:
    results: list[ToolResult]
    citations: dict[int, str]
    sources: dict[str, list[ToolResult]]
    errors: list[str]


def _evidence_index(state: AgentState) -> EvidenceIndex:
    results = [s["tool_result"] for s in state.get("agent_steps", []) if s["tool_result"] and s["tool_result"].ok]
    sources: dict[str, list[ToolResult]] = {}
    rows: list[Any] = list(state.get("citations", []))
    errors: list[str] = []
    for result in results:
        if not result.source_ids:
            errors.append("Successful evidence has no stable source IDs.")
        for sid in result.source_ids:
            sources.setdefault(sid, []).append(result)
        if isinstance(result.data, (dict, list)):
            result_sources = result.data.get("sources", []) if isinstance(result.data, dict) else result.data
            if not isinstance(result_sources, list):
                errors.append("Malformed citation sources.")
                continue
            for row in result_sources:
                if not isinstance(row, dict) or row.get("source_id") not in result.source_ids:
                    errors.append("Citation source is not linked to its tool result.")
                else:
                    rows.append(row)
    # Document identity must already exist; never guess provenance from a URL/title.
    for doc in state.get("documents", []):
        metadata = doc.get("metadata", {})
        rows.append({**metadata, **doc})
    candidates: dict[int, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("Malformed citation source.")
            continue
        source_id, cid = row.get("source_id"), row.get("citation_id")
        if not isinstance(source_id, str) or source_id not in sources:
            errors.append("Citation source is not linked to successful tool evidence.")
            continue
        if cid is None:  # Documents may supply content without a display citation.
            continue
        if type(cid) is not int or cid < 1:
            errors.append("Invalid citation ID.")
            continue
        candidates.setdefault(cid, set()).add(source_id)
    citations = {}
    for cid, ids in candidates.items():
        if len(ids) != 1:
            errors.append(f"Ambiguous citation [{cid}].")
        else:
            citations[cid] = next(iter(ids))
    return EvidenceIndex(results, citations, sources, list(dict.fromkeys(errors)))


def _retries(state: AgentState, failed: bool) -> int:
    return min(
        state.get("validation_retries", 0) + failed,
        state.get("max_validation_retries", MAX_VALIDATION_RETRIES),
    )


def validate_evidence_node(
    state: AgentState, tool_policy: ToolSelectionPolicy | None = None, tracer: LangSmithTracer | None = None
) -> dict[str, Any]:
    with (tracer or get_tracer()).trace(
        "validate_evidence",
        "chain",
        inputs={
            "query": state.get("query"),
            "agent_steps": state.get("agent_steps", []),
            "evidence_policy": state.get("evidence_policy"),
        },
    ) as span:
        index = _evidence_index(state)
        evidence = state.get("evidence_policy")
        successful = [s for s in state.get("agent_steps", []) if s["tool_result"] and s["tool_result"].ok]
        errors = list(index.errors)
        if evidence:
            missing = evidence.required_tools - {s["action"] for s in successful}
            if missing:
                errors.append("Missing successful tools: " + ", ".join(sorted(missing)))
            if evidence.required_authority and not any(
                r.source_authority == evidence.required_authority and r.source_ids for r in index.results
            ):
                errors.append(f"Missing {evidence.required_authority} evidence.")
            if any(s["action"] in evidence.forbidden_tools for s in successful):
                errors.append("Evidence was obtained using a forbidden tool.")
            if "calculator" in evidence.required_tools and not any(
                r.source_authority == "deterministic" and r.source_ids and _canonical(r.data) is not None
                for r in index.results
            ):
                errors.append("Missing deterministic calculation evidence.")
        if state.get("tool_results") and not successful:
            errors.append("No successful tool evidence is available.")
        # Requirements are supplied by the caller/classifier for this question.
        # A general information request must not require an entire business profile.
        required_fields = state.get("metadata", {}).get("required_business_fields", [])
        context = state.get("business_context", {})
        missing_info = [field for field in required_fields if context.get(field) in (None, "", [])]
        if missing_info:
            errors.append("Missing business facts: " + ", ".join(missing_info))
        previous = next((s["action"] for s in reversed(state.get("agent_steps", [])) if s["tool_result"]), None)
        if (tool_policy or ToolSelectionPolicy()).check_forbidden_transition(previous, "synthesize", evidence):
            errors.append("FORBIDDEN_TRANSITION: cannot synthesize after this evidence source.")
        updates = {
            "validation_errors": errors,
            "missing_info": missing_info
            + [error for error in errors if not error.startswith("Missing business facts:")],
            "validation_retries": _retries(state, bool(errors)),
            "agent_steps": [
                *state.get("agent_steps", []),
                step("validate_evidence", "; ".join(errors) or "Evidence requirements satisfied."),
            ],
        }
        span.set_outputs({"validation_errors": errors, "missing_info": updates["missing_info"]})
        return updates


def _canonical(value: Any) -> str | None:
    """Canonical decimal comparison, with no float tolerance or arithmetic inference."""
    if not isinstance(value, str) or not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value.strip()):
        return None
    try:
        number = Decimal(value.strip())
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number == 0:
        return "0"
    return format(number, "f").rstrip("0").rstrip(".") if "." in format(number, "f") else str(number)


def _verdict(
    claim: str,
    status: Literal["supported", "unsupported", "indeterminate"],
    method: Literal["deterministic", "citation_check", "authority_check", "llm_assisted"],
    ids: list[int],
    refs: list[str],
) -> ClaimVerdict:
    return {
        "claim": claim,
        "status": status,
        "method": method,
        "evidence_ids": ids,
        "tool_result_refs": sorted(set(refs)),
    }


def _claims(answer: str) -> list[str]:
    # Keep citations after punctuation attached to the preceding claim.
    answer = re.sub(r"([.!?])\s+(\[\d+\](?:\s*\[\d+\])*)", r" \2\1", answer)
    return [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", answer) if part.strip()]


def _non_claim(text: str) -> bool:
    clean = text.strip(" #*-\t")
    return bool(
        clean.endswith("?")
        or re.fullmatch(r"(?:hi|hello|hello again|thanks|thank you)[!.]?", clean, re.I)
        or re.fullmatch(r"Please (?:provide|clarify|share|confirm)\b[^.!?]*[.!?]?", clean, re.I)
    )


def _prose_verdicts(claims: list[str], index: EvidenceIndex, llm: BaseLLMService | None) -> list[ClaimVerdict]:
    verdicts = [
        _verdict(
            claim,
            "indeterminate",
            "llm_assisted",
            list(map(int, _CITATION.findall(claim))),
            [index.citations[int(cid)] for cid in _CITATION.findall(claim) if int(cid) in index.citations],
        )
        for claim in claims
    ]
    evidence = []
    for sid, results in index.sources.items():
        for result in results:
            if result.source_authority == "deterministic":
                continue
            # The tool's response is synthesis, not primary evidence. Only source
            # content/snippets and PDF document text can support semantic judgments.
            data = result.data if isinstance(result.data, dict) else {}
            result_sources = result.data if isinstance(result.data, list) else data.get("sources", [])
            rows = [*(result_sources if isinstance(result_sources, list) else []), data.get("document", {})]
            for row in rows:
                if not isinstance(row, dict) or row.get("source_id") != sid:
                    continue
                content = row.get("content") or row.get("text") or row.get("snippet")
                if content:
                    evidence.append({"source_id": sid, "content": str(content)[:8000]})
    if not claims or not evidence or llm is None:
        return verdicts
    try:
        # Isolate temperature from the shared synthesis service and concurrent turns.
        judge = copy(llm)
        judge.temperature = 0.0
        payload = {"claims": [{"id": i, "claim": c} for i, c in enumerate(claims)], "evidence": evidence}
        raw = judge.generate(json.dumps(payload), system=load_prompt("validation_claims"), max_tokens=2048)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("verdicts"), list):
            return verdicts
        rows = parsed["verdicts"]
        for i, verdict in enumerate(verdicts):
            matching = [r for r in rows if isinstance(r, dict) and type(r.get("id")) is int and r["id"] == i]
            if len(matching) != 1:
                continue
            row = matching[0]
            refs = row.get("source_ids")
            status = row.get("status")
            if (
                status not in {"supported", "unsupported", "indeterminate"}
                or not isinstance(refs, list)
                or not all(isinstance(s, str) for s in refs)
            ):
                continue
            allowed = {e["source_id"] for e in evidence}
            if verdict["evidence_ids"]:
                allowed &= {index.citations.get(cid) for cid in verdict["evidence_ids"]}
            if not refs or not set(refs) <= allowed:
                continue
            verdicts[i] = _verdict(verdict["claim"], status, "llm_assisted", verdict["evidence_ids"], refs)
    except Exception:
        # Unavailable or malformed semantic judgments cannot validate an answer.
        return verdicts
    return verdicts


def validate_claims_node(
    state: AgentState, llm: BaseLLMService | None = None, tracer: LangSmithTracer | None = None
) -> dict[str, Any]:
    with (tracer or get_tracer()).trace("validate_claims", "chain", inputs={"query": state.get("query")}) as span:
        index = _evidence_index(state)
        answer = validation_body(state.get("answer") or "")
        verdicts: list[ClaimVerdict] = []
        sources = [SourceInfo("", "", "", sid, cid, 0, 1) for cid, sid in index.citations.items()]
        for cid in dict.fromkeys(map(int, _CITATION.findall(answer))):
            valid = validate_citations(f"[{cid}]", sources)
            verdicts.append(
                _verdict(
                    f"Citation [{cid}] resolves to a retrieved source.",
                    "supported" if valid else "unsupported",
                    "citation_check",
                    [cid],
                    [index.citations[cid]] if valid else [],
                )
            )
        calculations = [
            r
            for r in index.results
            if r.source_authority == "deterministic" and r.source_ids and _canonical(r.data) is not None
        ]
        prose = []
        for claim in _claims(answer):
            if _non_claim(claim):
                continue
            ids = list(dict.fromkeys(map(int, _CITATION.findall(claim))))
            refs = [index.citations[cid] for cid in ids if cid in index.citations]
            legal = bool(_LEGAL.search(claim))
            clean = _LEGAL_NUMBER.sub("", _CITATION.sub("", claim)).replace("**", "")
            numbers = _NUMBER.findall(clean)
            policy = state.get("evidence_policy")
            numeric = bool(numbers) and (
                bool(calculations)
                or bool(policy and "calculator" in policy.required_tools)
                or bool(re.fullmatch(r"[\s₹$€£+\-\d,.]+", clean))
                or bool(re.search(r"\b(?:calculate\w*|total|liability|amount|equals)\b", clean, re.I))
            )
            if numeric:
                candidates = [r for r in calculations if not ids or set(r.source_ids) & set(refs)]
                for number in numbers:
                    value = _canonical(number.replace(",", ""))
                    matching = [r for r in candidates if _canonical(r.data) == value]
                    verdicts.append(
                        _verdict(
                            f"{claim} (numeric value: {number})",
                            "supported" if matching else ("unsupported" if candidates else "indeterminate"),
                            "deterministic",
                            ids,
                            [sid for r in (matching or candidates) for sid in r.source_ids],
                        )
                    )
            if legal:
                authoritative = [
                    sid
                    for sid in refs
                    if any(r.source_authority == "authoritative" for r in index.sources.get(sid, []))
                ]
                # Existing but weak citations are a policy violation; absent support
                # is unknown, not proof that a legal statement is false.
                status: Literal["supported", "unsupported", "indeterminate"] = (
                    "supported" if authoritative else ("unsupported" if refs else "indeterminate")
                )
                verdicts.append(_verdict(claim, status, "authority_check", ids, authoritative or refs))
            elif not numeric:
                prose.append(claim)
                if not ids and index.results:
                    verdicts.append(_verdict(claim, "indeterminate", "citation_check", [], []))
        verdicts.extend(_prose_verdicts(prose, index, llm))
        if not answer.strip():
            verdicts.append(_verdict("An answer is available.", "unsupported", "citation_check", [], []))
        unresolved = any(v["status"] != "supported" for v in verdicts)
        errors = list(index.errors)
        errors.extend(f"{v['status']}: {v['claim']}" for v in verdicts if v["status"] != "supported")
        failed = unresolved or bool(errors)
        updates: dict[str, Any] = {
            "claim_verdicts": verdicts,
            "validation_errors": errors,
            "validation_retries": _retries(state, failed),
            "metadata": {
                **state.get("metadata", {}),
                "agent_turn_completed": not failed,
                "validation_warnings": re.findall(r"\b(?:I assume|likely)\b", answer, re.I),
            },
            "agent_steps": [
                *state.get("agent_steps", []),
                step("validate_claims", "; ".join(errors) or "Available claim checks passed."),
            ],
        }
        if any(v["status"] == "unsupported" for v in verdicts):
            updates.update(citations_valid=False, confidence=0.4)
        elif failed:
            updates["confidence"] = 0.4
        else:
            updates.update(citations_valid=True, confidence=1.0)
        span.set_outputs({"claim_verdicts": verdicts, "validation_errors": errors})
        return updates

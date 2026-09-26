"""Deterministic golden checks of observed runs, independent of runtime policy."""

from __future__ import annotations

import itertools
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


def score_agent_case(case: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    steps = state.get("agent_steps", [])
    calls = [s for s in steps if s.get("tool_result") is not None]
    counts = Counter(s["action"] for s in calls)
    actual = set(counts)
    successful = [s for s in calls if s["tool_result"].ok]
    required = set(case.get("required_tools", []))
    recall = len(required & {s["action"] for s in successful}) / len(required) if required else 1.0
    forbidden = sorted(actual & set(case.get("forbidden_tools", [])))
    per_tool = [name for name, maximum in case.get("max_calls_per_tool", {}).items() if counts[name] > maximum]
    actions = [s["action"] for s in steps if s["action"] in actual | {"synthesize"}]
    transitions = [
        list(pair) for pair in itertools.pairwise(actions) if list(pair) in case.get("forbidden_transitions", [])
    ]
    results = [s["tool_result"] for s in successful]
    evidence = case.get("evidence_policy", {})
    authority = evidence.get("required_authority") or case.get("source_authority")
    authority_ok = authority is None or any(r.source_authority == authority and r.source_ids for r in results)
    evidence_tools_ok = set(evidence.get("required_tools", [])) <= {s["action"] for s in successful}
    forbidden = sorted(set(forbidden) | (actual & set(evidence.get("forbidden_tools", []))))
    expected = case.get("expected_calculation")
    calculation = (
        None
        if expected is None
        else any(r.source_authority == "deterministic" and r.data == expected and r.source_ids for r in results)
    )
    budget_ok = state.get("tool_calls_count", len(calls)) <= case.get("max_tool_calls", float("inf"))
    terminated = bool(state.get("metadata", {}).get("agent_turn_completed"))
    verdicts = state.get("claim_verdicts", [])
    statuses = Counter(v["status"] for v in verdicts)
    methods = Counter(v["method"] for v in verdicts)
    method_statuses: dict[str, dict[str, int]] = {}
    for verdict in verdicts:
        breakdown = method_statuses.setdefault(verdict["method"], {})
        breakdown[verdict["status"]] = breakdown.get(verdict["status"], 0) + 1
    known_citations = {c["citation_id"] for c in state.get("citations", []) if c.get("source_id")}
    cited = {int(cid) for cid in re.findall(r"\[(\d+)\]", state.get("answer") or "")}
    valid_cited = cited & known_citations
    expected_citations = case.get("expected_citations", 0)
    citation_coverage = len(valid_cited) / max(len(cited), expected_citations) if cited or expected_citations else None
    selection_ok = (
        recall == 1 and not forbidden and not per_tool and actual <= required | set(case.get("allowed_tools", []))
    )
    return {
        "tool_selection_correct": selection_ok,
        "claim_verdicts": verdicts,
        "claim_status_counts": {status: statuses[status] for status in ("supported", "unsupported", "indeterminate")},
        "claim_method_counts": dict(methods),
        "claim_method_status_counts": method_statuses,
        "citation_coverage": citation_coverage,
        "valid_citation_count": len(valid_cited),
        "invalid_citation_count": len(cited - known_citations),
        "expected_citations_met": len(valid_cited) >= expected_citations,
        "id": case["id"],
        "required_tools_recall": recall,
        "forbidden_violations": forbidden,
        "per_tool_violations": per_tool,
        "forbidden_transitions": transitions,
        "authority_compliant": authority_ok,
        "calculation_correct": calculation,
        "unnecessary_tool_calls": sum(counts[n] for n in actual - required - set(case.get("allowed_tools", []))),
        "policy_compliant": recall == 1
        and not forbidden
        and not per_tool
        and not transitions
        and authority_ok
        and evidence_tools_ok
        and budget_ok
        and calculation is not False,
        "iterations": state.get("iteration", 0),
        "tool_calls_count": state.get("tool_calls_count", 0),
        "tool_counts": dict(counts),
        "tool_errors": [
            {"tool": s["action"], "error_code": s["tool_result"].error_code} for s in calls if not s["tool_result"].ok
        ],
        "validation_retries": state.get("validation_retries", 0),
        "terminated": terminated,
        "finalized_with_limitations": state.get("finalized_with_limitations", False),
    }


def run_agent_eval(dataset_path: Path, limit: int | None = None, graph: Any = None) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    cases = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("Dataset contains no cases")
    if graph is None:
        from nagrik_ai.factories import create_agent_graph

        graph = create_agent_graph()
    scores = []
    for case in cases:
        start = time.perf_counter()
        # Golden requirements are only scored, never injected into runtime policy or budgets.
        error = None
        try:
            state = graph.invoke({"query": case["question"], "business_context": case.get("business_context", {})})
        except Exception as exc:
            state = {}
            error = type(exc).__name__
        score = score_agent_case(case, state)
        score["latency_ms"] = (time.perf_counter() - start) * 1000
        score["error"] = error
        score["policy_compliant"] = score["policy_compliant"] and error is None
        scores.append(score)
    n = len(scores)
    status_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    method_status_counts: dict[str, Counter[str]] = {}
    for score in scores:
        status_counts.update(score["claim_status_counts"])
        method_counts.update(score["claim_method_counts"])
        for method, counts in score["claim_method_status_counts"].items():
            method_status_counts.setdefault(method, Counter()).update(counts)
    claim_count = sum(status_counts.values())
    coverage = [s["citation_coverage"] for s in scores if s["citation_coverage"] is not None]
    summary = {
        "tool_selection_accuracy": sum(s["tool_selection_correct"] and s["error"] is None for s in scores) / n,
        "authority_compliance_rate": sum(s["authority_compliant"] and s["error"] is None for s in scores) / n,
        "claim_status_counts": dict(status_counts),
        "claim_method_counts": dict(method_counts),
        "claim_method_status_counts": {method: dict(counts) for method, counts in method_status_counts.items()},
        "indeterminate_rate": status_counts["indeterminate"] / claim_count if claim_count else 0.0,
        "supported_rate": status_counts["supported"] / claim_count if claim_count else 0.0,
        "unsupported_rate": status_counts["unsupported"] / claim_count if claim_count else 0.0,
        "claims_evaluated": claim_count,
        "citation_coverage": sum(coverage) / len(coverage) if coverage else None,
        "citation_cases_evaluated": len(coverage),
        "cases": n,
        "required_tools_recall": sum(s["required_tools_recall"] for s in scores) / n,
        "policy_compliance_rate": sum(s["policy_compliant"] for s in scores) / n,
        "forbidden_violations": sum(len(s["forbidden_violations"]) for s in scores),
        "per_tool_violations": sum(len(s["per_tool_violations"]) for s in scores),
        "forbidden_transition_violations": sum(len(s["forbidden_transitions"]) for s in scores),
        "tool_calls_count": sum(s["tool_calls_count"] for s in scores),
        "termination_rate": sum(s["terminated"] for s in scores) / n,
        "limitations_rate": sum(s["finalized_with_limitations"] for s in scores) / n,
        "execution_errors": sum(s["error"] is not None for s in scores),
    }
    return {"summary": summary, "cases": scores}

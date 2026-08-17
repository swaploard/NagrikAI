"""Write RAG evaluation runs to JSON and Markdown reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from evaluation.runners.rag_runner import RagCaseResult, RagEvalResult


def write_report(result: RagEvalResult, output_dir: str | Path) -> tuple[Path, Path]:
    """Write ``report_<timestamp>.json`` and ``summary_<timestamp>.md`` into ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = out / f"report_{timestamp}.json"
    md_path = out / f"summary_{timestamp}.md"
    json_path.write_text(_to_json(result), encoding="utf-8")
    md_path.write_text(_to_markdown(result), encoding="utf-8")
    return json_path, md_path


def aggregate_metrics(result: RagEvalResult) -> dict[str, dict[str, float]]:
    """Per-metric aggregates: mean score, pass rate, and number of scored cases."""
    aggregates: dict[str, list[float]] = {}
    passed: dict[str, list[bool]] = {}
    for case in result.cases:
        for score in case.metric_scores:
            if score.score is None:
                continue
            aggregates.setdefault(score.metric, []).append(score.score)
            passed.setdefault(score.metric, []).append(score.success)
    summary: dict[str, dict[str, float]] = {}
    for name, values in aggregates.items():
        successes = passed.get(name, [])
        summary[name] = {
            "mean": fmean(values),
            "min": min(values),
            "max": max(values),
            "pass_rate": (sum(1 for ok in successes if ok) / len(successes)) if successes else 0.0,
            "cases": float(len(values)),
        }
    return summary


def mean_overall_score(result: RagEvalResult) -> float:
    """Mean across every scored metric; 0.0 when no scores exist."""
    scores = [score.score for case in result.cases for score in case.metric_scores if score.score is not None]
    return fmean(scores) if scores else 0.0


def _to_json(result: RagEvalResult) -> str:
    payload: dict[str, object] = {
        "test_run_id": result.test_run_id,
        "judge_model": result.judge_model,
        "offline": result.offline,
        "aggregate": aggregate_metrics(result),
        "mean_overall_score": mean_overall_score(result),
        "cases": [
            {
                "case_id": case.case_id,
                "input": case.input,
                "actual_output": case.actual_output,
                "expected_output": case.expected_output,
                "latency_ms": case.latency_ms,
                "retrieval_metrics": case.retrieval_metrics,
                "metric_scores": [
                    {
                        "metric": score.metric,
                        "score": score.score,
                        "threshold": score.threshold,
                        "success": score.success,
                        "reason": score.reason,
                    }
                    for score in case.metric_scores
                ],
            }
            for case in result.cases
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _to_markdown(result: RagEvalResult) -> str:
    lines = [
        "# NagrikAI DeepEval Report",
        "",
        f"- Judge model: `{result.judge_model}`",
        f"- Test run ID: `{result.test_run_id or 'n/a (offline)'}`",
        f"- Mode: {'offline (no LLM judge)' if result.offline else 'online'}",
        f"- Cases: {len(result.cases)}",
        "",
        "## Aggregate",
        "",
        _aggregate_table(result),
        "",
        "## Per-case",
        "",
    ]
    for case in result.cases:
        lines.extend(_case_block(case))
    return "\n".join(lines)


def _aggregate_table(result: RagEvalResult) -> str:
    aggregates = aggregate_metrics(result)
    rows = ["| Metric | Mean | Min | Max | Pass rate | Cases |", "|---|---|---|---|---|---|"]
    for name in sorted(aggregates):
        agg = aggregates[name]
        rows.append(
            f"| {name} | {agg['mean']:.3f} | {agg['min']:.3f} | {agg['max']:.3f} "
            f"| {agg['pass_rate']:.0%} | {agg['cases']:.0f} |"
        )
    return "\n".join(rows)


def _case_block(case: RagCaseResult) -> list[str]:
    lines = [
        f"### {case.case_id}",
        "",
        f"- **Latency:** {case.latency_ms:.1f} ms",
        f"- **Input:** {case.input}",
    ]
    if case.metric_scores:
        lines.extend(
            [
                "",
                "| Metric | Score | Threshold | Pass | Reason |",
                "|---|---|---|---|---|",
            ]
        )
        for score in case.metric_scores:
            score_text = f"{score.score:.3f}" if score.score is not None else "n/a"
            threshold_text = f"{score.threshold:.3f}" if score.threshold is not None else "n/a"
            lines.append(
                f"| {score.metric} | {score_text} | {threshold_text} | "
                f"{'yes' if score.success else 'no'} | {score.reason.replace('|', '\\|')} |"
            )
    return lines

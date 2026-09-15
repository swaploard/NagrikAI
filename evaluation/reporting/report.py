"""Write RAG evaluation runs to JSON and Markdown reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from evaluation.runners.rag_runner import RagCaseResult, RagEvalResult, aggregate_retrieval_ranking


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


def aggregate_ranking_metrics(result: RagEvalResult) -> dict[str, dict[str, float]]:
    """Per-ranking-metric aggregates: mean, min, max, and number of cases."""
    values_by_metric: dict[str, list[float]] = {}
    for case in result.cases:
        for name, value in case.ranking_metrics.items():
            if name == "ap" and case.ranking_metrics.get("num_relevant", 0.0) <= 0.0:
                continue
            output_name = "map" if name == "ap" else "mrr" if name == "rr" else name
            values_by_metric.setdefault(output_name, []).append(value)
    return {
        name: {
            "mean": fmean(values),
            "min": min(values),
            "max": max(values),
            "cases": float(len(values)),
        }
        for name, values in values_by_metric.items()
        if values
    }


def _to_json(result: RagEvalResult) -> str:
    ranking_summary = aggregate_retrieval_ranking(result)
    payload: dict[str, object] = {
        "test_run_id": result.test_run_id,
        "judge_model": result.judge_model,
        "offline": result.offline,
        "aggregate": aggregate_metrics(result),
        "aggregate_ranking": aggregate_ranking_metrics(result),
        "mean_map": ranking_summary.get("map", 0.0),
        "mean_mrr": ranking_summary.get("mrr", 0.0),
        "mean_overall_score": mean_overall_score(result),
        "cases": [
            {
                "case_id": case.case_id,
                "input": case.input,
                "actual_output": case.actual_output,
                "expected_output": case.expected_output,
                "latency_ms": case.latency_ms,
                "retrieval_metrics": case.retrieval_metrics,
                "ranking_metrics": case.ranking_metrics,
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
    rows.extend(["", "### Retrieval Ranking (mean over cases)", ""])
    ranking = aggregate_ranking_metrics(result)
    ranking_rows = ["| Metric | Mean | Min | Max | Cases |", "|---|---|---|---|---|"]
    preferred = [
        "map",
        "mrr",
        "ndcg@1",
        "ndcg@3",
        "ndcg@5",
        "ndcg@10",
        "precision@1",
        "precision@3",
        "precision@5",
        "precision@10",
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
    ]
    for name in [metric for metric in preferred if metric in ranking]:
        agg = ranking[name]
        ranking_rows.append(
            f"| {name} | {agg['mean']:.3f} | {agg['min']:.3f} | {agg['max']:.3f} | {agg['cases']:.0f} |"
        )
    rows.extend(ranking_rows)
    return "\n".join(rows)


def _case_block(case: RagCaseResult) -> list[str]:
    lines = [
        f"### {case.case_id}",
        "",
        f"- **Latency:** {case.latency_ms:.1f} ms",
        f"- **Input:** {case.input}",
    ]
    if case.ranking_metrics:
        lines.extend(
            [
                "",
                "| AP | RR | nDCG@5 | P@5 | R@5 |",
                "|---|---|---|---|---|",
                (
                    f"| {case.ranking_metrics.get('ap', 0.0):.3f} | "
                    f"{case.ranking_metrics.get('rr', 0.0):.3f} | "
                    f"{case.ranking_metrics.get('ndcg@5', 0.0):.3f} | "
                    f"{case.ranking_metrics.get('precision@5', 0.0):.3f} | "
                    f"{case.ranking_metrics.get('recall@5', 0.0):.3f} |"
                ),
            ]
        )
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

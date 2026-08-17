"""CLI for the NagrikAI evaluation suite (`nagrik-eval`)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from evaluation.datasets.loader import GoldenDatasetError
from evaluation.judges.factory import JudgeConfigError, create_eval_judge
from evaluation.reporting.report import mean_overall_score, write_report
from evaluation.runners.rag_runner import build_default_metrics, run_rag_eval

DEFAULT_DATASET = Path("evaluation/datasets/rag/golden_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("evaluation/results")

app = typer.Typer(name="nagrik-eval", help="NagrikAI evaluation suite")
deepeval_app = typer.Typer(name="deepeval", help="DeepEval-powered RAG evaluation")
app.add_typer(deepeval_app, help="DeepEval-powered evaluation commands")


@deepeval_app.command("run")
def deepeval_run(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="Path to the golden dataset JSONL")] = (
        DEFAULT_DATASET
    ),
    limit: Annotated[int | None, typer.Option("--limit", help="Maximum number of cases to evaluate")] = None,
    offset: Annotated[int, typer.Option("--offset", help="Skip the first N cases")] = 0,
    metric: Annotated[
        list[str] | None, typer.Option("--metric", "-m", help="Metric name(s) to run (repeatable)")
    ] = None,
    threshold: Annotated[float, typer.Option("--threshold", help="Minimum mean score; exit 1 below it")] = 0.0,
    offline: Annotated[bool, typer.Option("--offline", help="Skip LLM judge metrics (latency + retrieval only)")] = (
        False
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o", help="Report output directory")] = (
        DEFAULT_OUTPUT_DIR
    ),
    judge_provider: Annotated[
        str | None, typer.Option("--judge-provider", help="Judge provider: ollama or openrouter")
    ] = None,
    judge_model: Annotated[str | None, typer.Option("--judge-model", help="Judge model override")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Run the RAG pipeline over a golden dataset and score it with DeepEval."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        judge = create_eval_judge(provider=judge_provider, model=judge_model)
    except JudgeConfigError as exc:
        typer.echo(f"Judge configuration error: {exc}", err=True)
        raise typer.Exit(1) from exc

    metrics = build_default_metrics(judge, include=set(metric) if metric else None)
    typer.echo(f"Judge model: {judge.get_model_name()}")
    typer.echo(
        f"Running {'offline' if offline else 'online'} eval on {dataset} "
        f"({len(metrics)} metric(s): {', '.join(m.__name__ for m in metrics)})"
    )

    try:
        result = run_rag_eval(
            dataset_path=dataset,
            judge=judge,
            metrics=metrics if not offline else None,
            limit=limit,
            offset=offset,
            offline=offline,
        )
    except (GoldenDatasetError, RuntimeError) as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    json_path, md_path = write_report(result, output_dir)
    typer.echo(f"Report written: {json_path}")
    typer.echo(f"Summary written: {md_path}")

    overall = mean_overall_score(result)
    if offline:
        typer.echo(f"Offline run: {len(result.cases)} cases, no judge scores.")
        return

    typer.echo(f"Mean overall score: {overall:.3f}")
    if overall < threshold:
        typer.echo(f"Mean score {overall:.3f} is below threshold {threshold:.3f}", err=True)
        raise typer.Exit(1)


@app.command("validate-datasets")
def validate_datasets(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="Path to the JSONL dataset")] = (DEFAULT_DATASET),
) -> None:
    """Validate that a golden dataset loads and conforms to the expected schema."""
    from evaluation.datasets.loader import load_golden_dataset

    try:
        goldens = load_golden_dataset(dataset)
    except GoldenDatasetError as exc:
        typer.echo(f"Dataset invalid: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Dataset OK: {len(goldens)} cases loaded from {dataset}")


if __name__ == "__main__":
    app()

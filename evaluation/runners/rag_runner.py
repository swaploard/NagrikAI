"""RAG-layer DeepEval runner: run the pipeline and score responses with DeepEval metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.evaluate.types import EvaluationResult, TestResult
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, RetrievedContextData, SingleTurnParams

from evaluation.datasets.loader import GoldenDatasetError, load_golden_dataset
from evaluation.rubrics.loader import load_geval_criteria
from nagrik_ai.agent.rag_graph import run_rag_query
from nagrik_ai.factories import create_retrieval_service
from nagrik_ai.services.document_retrieval_service import (
    DocumentRetrievalService,
    retrieval_metrics,
)

RUBRICS_DIR = Path(__file__).resolve().parent.parent / "rubrics"

DEFAULT_METRIC_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.5,
    "answer_relevancy": 0.5,
    "contextual_relevancy": 0.5,
    "contextual_precision": 0.5,
    "contextual_recall": 0.5,
    "hallucination": 0.5,
    "correctness": 0.5,
}


@dataclass
class MetricScore:
    """Per-case score for a single DeepEval metric."""

    metric: str
    score: float | None
    threshold: float | None
    success: bool
    reason: str = ""


@dataclass
class RagCaseResult:
    """Per-case outcome of a RAG evaluation run."""

    case_id: str
    input: str
    actual_output: str
    expected_output: str
    latency_ms: float
    retrieval_metrics: dict[str, Any]
    retrieval_context: list[RetrievedContextData | str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    metric_scores: list[MetricScore] = field(default_factory=list)


@dataclass
class RagEvalResult:
    """Full outcome of a RAG evaluation run."""

    cases: list[RagCaseResult]
    test_run_id: str | None
    judge_model: str
    offline: bool


def build_default_metrics(
    judge: DeepEvalBaseLLM,
    thresholds: dict[str, float] | None = None,
    include: set[str] | None = None,
) -> list[BaseMetric]:
    """Build the default DeepEval metric set for the RAG layer.

    ``include`` filters by metric name; ``thresholds`` overrides the default
    per-metric pass thresholds. All metrics run synchronously.
    """
    merged = {**DEFAULT_METRIC_THRESHOLDS, **(thresholds or {})}

    def _threshold(name: str) -> float:
        return merged[name]

    def _wanted(name: str) -> bool:
        return include is None or name in include

    metrics: list[BaseMetric] = []
    if _wanted("faithfulness"):
        metrics.append(FaithfulnessMetric(model=judge, async_mode=False, threshold=_threshold("faithfulness")))
    if _wanted("answer_relevancy"):
        metrics.append(AnswerRelevancyMetric(model=judge, async_mode=False, threshold=_threshold("answer_relevancy")))
    if _wanted("contextual_relevancy"):
        metrics.append(
            ContextualRelevancyMetric(model=judge, async_mode=False, threshold=_threshold("contextual_relevancy"))
        )
    if _wanted("contextual_precision"):
        metrics.append(
            ContextualPrecisionMetric(model=judge, async_mode=False, threshold=_threshold("contextual_precision"))
        )
    if _wanted("contextual_recall"):
        metrics.append(ContextualRecallMetric(model=judge, async_mode=False, threshold=_threshold("contextual_recall")))
    if _wanted("hallucination"):
        metrics.append(HallucinationMetric(model=judge, async_mode=False, threshold=_threshold("hallucination")))
    if _wanted("correctness"):
        criteria = load_geval_criteria(RUBRICS_DIR / "correctness.md")
        metrics.append(
            GEval(
                name="Correctness",
                criteria=criteria,
                evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
                model=judge,
                async_mode=False,
                threshold=_threshold("correctness"),
            )
        )
    return metrics


def run_rag_eval(
    dataset_path: str | Path,
    judge: DeepEvalBaseLLM,
    metrics: list[BaseMetric] | None = None,
    limit: int | None = None,
    offset: int = 0,
    offline: bool = False,
    retrieval_service: DocumentRetrievalService | None = None,
) -> RagEvalResult:
    """Run the RAG pipeline over a golden dataset and score it with DeepEval.

    In offline mode the pipeline still runs (for latency and deterministic
    retrieval metrics) but no LLM judge metrics are scored.
    """
    goldens = load_golden_dataset(dataset_path)
    selected = goldens[offset:] if limit is None else goldens[offset : offset + limit]
    if not selected:
        raise GoldenDatasetError(f"No cases selected from {dataset_path} (offset={offset}, limit={limit})")
    service = retrieval_service or create_retrieval_service()

    cases: list[RagCaseResult] = []
    for golden in selected:
        query = golden.input
        expected = golden.expected_output
        if not isinstance(query, str) or not query.strip():
            raise GoldenDatasetError(f"Golden {golden.id!r} has no input")
        if not isinstance(expected, str) or not expected.strip():
            raise GoldenDatasetError(f"Golden {golden.id!r} has no expected_output")

        docs = service.retrieve(query)
        deterministic = retrieval_metrics(docs)
        result = run_rag_query(query, retrieval_service=service)
        cases.append(
            RagCaseResult(
                case_id=golden.id or "",
                input=query,
                actual_output=result.response,
                expected_output=expected,
                latency_ms=result.latency_ms,
                retrieval_metrics=deterministic,
                retrieval_context=cast("list[RetrievedContextData | str]", result.raw_chunks),
                context=result.raw_chunks,
            )
        )

    if offline:
        return RagEvalResult(cases=cases, test_run_id=None, judge_model=judge.get_model_name(), offline=True)

    test_cases = [
        LLMTestCase(
            input=case.input,
            actual_output=case.actual_output,
            expected_output=case.expected_output,
            context=case.context,
            retrieval_context=case.retrieval_context,
        )
        for case in cases
    ]
    eval_result = evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(run_async=False, max_concurrent=1),
        display_config=DisplayConfig(show_indicator=True, print_results=True),
    )
    _attach_scores(eval_result, cases)
    return RagEvalResult(
        cases=cases,
        test_run_id=eval_result.test_run_id,
        judge_model=judge.get_model_name(),
        offline=False,
    )


def _attach_scores(eval_result: EvaluationResult, cases: list[RagCaseResult]) -> None:
    by_index: dict[int, TestResult] = {
        test_result.index: test_result for test_result in eval_result.test_results if test_result.index is not None
    }
    for index, case in enumerate(cases):
        test_result = by_index.get(index)
        if test_result is None or test_result.metrics_data is None:
            continue
        for metric_data in test_result.metrics_data:
            case.metric_scores.append(
                MetricScore(
                    metric=metric_data.name,
                    score=metric_data.score,
                    threshold=metric_data.threshold,
                    success=bool(metric_data.success),
                    reason=metric_data.reason or "",
                )
            )

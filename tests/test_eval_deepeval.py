from __future__ import annotations

import json
from pathlib import Path

import pytest
from deepeval.dataset import Golden
from deepeval.models import OllamaModel, OpenRouterModel

from evaluation.datasets.loader import GoldenDatasetError, load_golden_dataset
from evaluation.judges.factory import JudgeConfigError, create_eval_judge
from evaluation.reporting.report import aggregate_metrics, mean_overall_score
from evaluation.rubrics.loader import load_geval_criteria
from evaluation.runners.rag_runner import (
    DEFAULT_METRIC_THRESHOLDS,
    MetricScore,
    RagCaseResult,
    RagEvalResult,
    build_default_metrics,
)

GOLDEN_DATASET = Path("evaluation/datasets/rag/golden_dataset.jsonl")
RUBRIC_CORRECTNESS = Path("evaluation/rubrics/correctness.md")


class TestGoldenDatasetLoader:
    def test_loads_all_records(self) -> None:
        goldens = load_golden_dataset(GOLDEN_DATASET)
        line_count = len([line for line in GOLDEN_DATASET.read_text(encoding="utf-8").splitlines() if line.strip()])
        assert len(goldens) == line_count
        for golden in goldens:
            assert isinstance(golden, Golden)
            assert isinstance(golden.input, str) and golden.input
            assert isinstance(golden.expected_output, str) and golden.expected_output
            assert golden.retrieval_context, "each record must have retrieval_context"

    def test_ids_are_unique(self) -> None:
        goldens = load_golden_dataset(GOLDEN_DATASET)
        ids = [golden.id for golden in goldens]
        assert len(ids) == len(set(ids))

    def test_metadata_preserved(self) -> None:
        goldens = load_golden_dataset(GOLDEN_DATASET)
        metadata = goldens[0].additional_metadata or {}
        assert metadata["question_type"] == "eligibility"
        assert metadata["verbosity"] == "concise"
        assert metadata["expected_citations"] == ["1", "2"]

    def test_missing_question_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({"id": "x", "expected_answer": "a"}), encoding="utf-8")
        with pytest.raises(GoldenDatasetError, match="question"):
            load_golden_dataset(bad)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(GoldenDatasetError, match="not found"):
            load_golden_dataset(tmp_path / "nope.jsonl")

    def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(GoldenDatasetError, match="empty"):
            load_golden_dataset(empty)


class TestJudgeFactory:
    def test_ollama_provider(self) -> None:
        judge = create_eval_judge(provider="ollama", model="qwen2.5:7b")
        assert isinstance(judge, OllamaModel)

    def test_openrouter_provider(self) -> None:
        judge = create_eval_judge(provider="openrouter", model="anthropic/claude-3.5-sonnet", api_key="test-key")
        assert isinstance(judge, OpenRouterModel)

    def test_unsupported_provider_raises(self) -> None:
        with pytest.raises(JudgeConfigError, match="Unsupported judge provider"):
            create_eval_judge(provider="unknown")

    def test_model_name_exposed(self) -> None:
        judge = create_eval_judge(provider="ollama", model="qwen2.5:7b")
        assert "qwen2.5:7b" in judge.get_model_name()


class TestRubricLoader:
    def test_criteria_contains_grading_guidance(self) -> None:
        criteria = load_geval_criteria(RUBRIC_CORRECTNESS)
        assert "Score 5" in criteria
        assert "factual_correctness_score" not in criteria, "JSON block must be stripped"
        assert "#" not in criteria.split("\n")[0] if criteria else True


class TestDefaultMetrics:
    def test_all_default_metrics_built(self) -> None:
        judge = create_eval_judge(provider="ollama", model="qwen2.5:7b")
        metrics = build_default_metrics(judge)
        names = {metric.__class__.__name__ for metric in metrics}
        assert names == {
            "FaithfulnessMetric",
            "AnswerRelevancyMetric",
            "ContextualRelevancyMetric",
            "ContextualPrecisionMetric",
            "ContextualRecallMetric",
            "HallucinationMetric",
            "GEval",
        }

    def test_include_filter(self) -> None:
        judge = create_eval_judge(provider="ollama", model="qwen2.5:7b")
        metrics = build_default_metrics(judge, include={"faithfulness", "correctness"})
        names = {metric.__class__.__name__ for metric in metrics}
        assert names == {"FaithfulnessMetric", "GEval"}

    def test_threshold_override(self) -> None:
        judge = create_eval_judge(provider="ollama", model="qwen2.5:7b")
        metrics = build_default_metrics(judge, thresholds={"faithfulness": 0.9})
        faithfulness = next(m for m in metrics if m.__class__.__name__ == "FaithfulnessMetric")
        assert faithfulness.threshold == 0.9
        assert DEFAULT_METRIC_THRESHOLDS["faithfulness"] == 0.5


class TestReportAggregates:
    def _sample_result(self) -> RagEvalResult:
        case_a = RagCaseResult(
            case_id="a",
            input="q",
            actual_output="o",
            expected_output="e",
            latency_ms=10.0,
            retrieval_metrics={},
            metric_scores=[
                MetricScore(metric="faithfulness", score=0.8, threshold=0.5, success=True, reason="ok"),
                MetricScore(metric="answer_relevancy", score=0.4, threshold=0.5, success=False, reason="bad"),
            ],
        )
        case_b = RagCaseResult(
            case_id="b",
            input="q2",
            actual_output="o2",
            expected_output="e2",
            latency_ms=20.0,
            retrieval_metrics={},
            metric_scores=[
                MetricScore(metric="faithfulness", score=0.6, threshold=0.5, success=True, reason="ok"),
            ],
        )
        return RagEvalResult(cases=[case_a, case_b], test_run_id="run-1", judge_model="j", offline=False)

    def test_aggregate_means_and_pass_rates(self) -> None:
        result = self._sample_result()
        aggregates = aggregate_metrics(result)
        assert aggregates["faithfulness"]["mean"] == pytest.approx(0.7)
        assert aggregates["faithfulness"]["pass_rate"] == pytest.approx(1.0)
        assert aggregates["answer_relevancy"]["mean"] == pytest.approx(0.4)
        assert aggregates["answer_relevancy"]["pass_rate"] == pytest.approx(0.0)

    def test_mean_overall_score(self) -> None:
        result = self._sample_result()
        assert mean_overall_score(result) == pytest.approx(0.6)

    def test_mean_overall_score_empty(self) -> None:
        result = RagEvalResult(cases=[], test_run_id=None, judge_model="j", offline=True)
        assert mean_overall_score(result) == 0.0

"""Load golden datasets (JSONL) into DeepEval Golden objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

from deepeval.dataset import Golden
from deepeval.test_case import RetrievedContextData


class GoldenDatasetError(RuntimeError):
    """Raised when a golden dataset file is malformed or missing."""


def load_golden_dataset(path: str | Path) -> list[Golden]:
    """Load a JSONL golden dataset into a list of DeepEval Goldens.

    Expected record shape (evaluation/datasets/rag/golden_dataset.jsonl):
    ``question``, ``expected_answer``, ``retrieved_documents[].relevant_text``,
    plus optional ``id``, ``question_type``, ``verbosity``, ``expected_citations``.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise GoldenDatasetError(f"Dataset not found: {dataset_path}")
    goldens: list[Golden] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldenDatasetError(f"{dataset_path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise GoldenDatasetError(f"{dataset_path}:{line_no}: record must be a JSON object")
            goldens.append(_record_to_golden(record, source_file=str(dataset_path), line_no=line_no))
    if not goldens:
        raise GoldenDatasetError(f"Dataset is empty: {dataset_path}")
    return goldens


def _record_to_golden(record: dict[str, Any], source_file: str, line_no: int) -> Golden:
    def _fail(message: str) -> NoReturn:
        raise GoldenDatasetError(f"{source_file}:{line_no}: {message}")

    question = record.get("question")
    expected_answer = record.get("expected_answer")
    if not isinstance(question, str) or not question.strip():
        _fail("missing non-empty 'question'")
    if not isinstance(expected_answer, str) or not expected_answer.strip():
        _fail("missing non-empty 'expected_answer'")

    retrieved = record.get("retrieved_documents")
    if not isinstance(retrieved, list):
        _fail("'retrieved_documents' must be a list")
    retrieval_context: list[RetrievedContextData | str] = [
        str(doc["relevant_text"]) for doc in retrieved if isinstance(doc, dict) and doc.get("relevant_text")
    ]

    case_id = record.get("id")
    if not isinstance(case_id, str) or not case_id:
        _fail("missing non-empty 'id'")

    metadata: dict[str, Any] = {}
    for key in ("question_type", "verbosity", "expected_citations", "expected_authorities"):
        if key in record:
            metadata[key] = record[key]

    return Golden(
        id=case_id,
        input=question,
        expected_output=expected_answer,
        retrieval_context=retrieval_context,
        additional_metadata=metadata,
        multimodal=False,
    )

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from nagrik_ai.agent.verbosity import VERBOSITY_CONCISE
from nagrik_ai.models.rag_result import SourceInfo
from nagrik_ai.services.citation_service import validate_citations

CONCISE_RESPONSE_WORD_CAP = 500


class ValidationStatus(StrEnum):
    PASS = "pass"  # noqa: S105 - validation status, not a credential
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class ValidationResult:
    status: ValidationStatus
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    retryable: bool = False


@dataclass(frozen=True)
class ValidationSummary:
    status: ValidationStatus
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    retryable: bool


class BaseValidator(Protocol):
    def validate(
        self,
        *,
        response: str,
        body: str,
        sources: list[SourceInfo],
        cited_ids: list[int],
        metadata: dict[str, object],
    ) -> ValidationResult: ...


class CitationValidator:
    def validate(
        self,
        *,
        response: str,
        body: str,
        sources: list[SourceInfo],
        cited_ids: list[int],
        metadata: dict[str, object],
    ) -> ValidationResult:
        del response, metadata
        not_found_patterns = [
            "could not find",
            "not available",
            "not found",
            "not covered",
            "does not contain",
            "do not contain",
            "no information",
            "unable to find",
        ]
        body_lower = body.lower()
        if not sources:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                errors=("No citation sources available",),
                retryable=False,
            )
        if any(pattern in body_lower for pattern in not_found_patterns):
            return ValidationResult(
                status=ValidationStatus.FAIL,
                errors=("LLM indicated information not found in sources",),
                retryable=False,
            )
        if not cited_ids:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                errors=("No inline citations in response body",),
                retryable=True,
            )
        if not validate_citations(body, sources):
            return ValidationResult(
                status=ValidationStatus.FAIL,
                errors=("Response contains citations not found in sources",),
                retryable=True,
            )
        return ValidationResult(status=ValidationStatus.PASS)


class FormattingValidator:
    def validate(
        self,
        *,
        response: str,
        body: str,
        sources: list[SourceInfo],
        cited_ids: list[int],
        metadata: dict[str, object],
    ) -> ValidationResult:
        del body, sources, cited_ids, metadata
        malformed = re.search(r"\[\s+\d+\s*\]|\[\s*\d+\s+\]", response)
        if malformed:
            return ValidationResult(
                status=ValidationStatus.WARN,
                warnings=("Citation spacing should use compact [n] format",),
                retryable=False,
            )
        return ValidationResult(status=ValidationStatus.PASS)


class LengthValidator:
    def validate(
        self,
        *,
        response: str,
        body: str,
        sources: list[SourceInfo],
        cited_ids: list[int],
        metadata: dict[str, object],
    ) -> ValidationResult:
        del response, sources, cited_ids
        verbosity = str(metadata.get("verbosity", VERBOSITY_CONCISE))
        response_word_count = len(body.split())
        if verbosity == VERBOSITY_CONCISE and response_word_count > CONCISE_RESPONSE_WORD_CAP:
            return ValidationResult(
                status=ValidationStatus.WARN,
                warnings=(f"Concise response has {response_word_count} words",),
                retryable=False,
            )
        return ValidationResult(status=ValidationStatus.PASS)


class TruncationValidator:
    def validate(
        self,
        *,
        response: str,
        body: str,
        sources: list[SourceInfo],
        cited_ids: list[int],
        metadata: dict[str, object],
    ) -> ValidationResult:
        del response, body, sources, cited_ids
        if metadata.get("finish_reason") == "length":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                errors=("LLM response truncated by token cap",),
                retryable=True,
            )
        return ValidationResult(status=ValidationStatus.PASS)


DEFAULT_VALIDATORS: tuple[BaseValidator, ...] = (
    CitationValidator(),
    FormattingValidator(),
    LengthValidator(),
    TruncationValidator(),
)


def validation_body(response: str) -> str:
    return re.split(r"\n\s*(?:Sources|References|स्रोत)\s*:?\s*\n", response, maxsplit=1)[0]


def cited_ids(response_body: str) -> list[int]:
    return [int(match) for match in re.findall(r"\[(\d+)\]", response_body)]


def validate_response(
    *,
    response: str,
    sources: list[SourceInfo],
    metadata: dict[str, object],
    validators: tuple[BaseValidator, ...] = DEFAULT_VALIDATORS,
) -> tuple[ValidationSummary, str, list[int]]:
    body = validation_body(response)
    ids = cited_ids(body)
    results = [
        validator.validate(
            response=response,
            body=body,
            sources=sources,
            cited_ids=ids,
            metadata=metadata,
        )
        for validator in validators
    ]

    errors = tuple(error for result in results for error in result.errors)
    warnings = tuple(warning for result in results for warning in result.warnings)
    retryable = any(result.retryable for result in results)
    if any(result.status == ValidationStatus.FAIL for result in results):
        status = ValidationStatus.FAIL
    elif any(result.status == ValidationStatus.WARN for result in results):
        status = ValidationStatus.WARN
    else:
        status = ValidationStatus.PASS

    return (
        ValidationSummary(
            status=status,
            warnings=warnings,
            errors=errors,
            retryable=retryable,
        ),
        body,
        ids,
    )

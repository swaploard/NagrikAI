"""Deterministic query classifiers (verbosity + question type). No LLM calls."""

from __future__ import annotations

import re
from typing import Literal

Verbosity = Literal["detailed", "concise"]

VERBOSITY_DETAILED: Verbosity = "detailed"
VERBOSITY_CONCISE: Verbosity = "concise"

QuestionType = Literal[
    "FACTUAL",
    "DEFINITION",
    "PROCEDURAL",
    "COMPARISON",
    "CALCULATION",
    "LEGAL_INTERPRETATION",
]

QUESTION_TYPE_FACTUAL: QuestionType = "FACTUAL"
QUESTION_TYPE_DEFINITION: QuestionType = "DEFINITION"
QUESTION_TYPE_PROCEDURAL: QuestionType = "PROCEDURAL"
QUESTION_TYPE_COMPARISON: QuestionType = "COMPARISON"
QUESTION_TYPE_CALCULATION: QuestionType = "CALCULATION"
QUESTION_TYPE_LEGAL_INTERPRETATION: QuestionType = "LEGAL_INTERPRETATION"

DETAILED_KEYWORDS: frozenset[str] = frozenset(
    {
        "compare",
        "difference",
        "analyse",
        "analyze",
        "why",
        "how",
        "procedure",
        "step",
        "calculation",
        "basis",
        "interpret",
        "example",
        "explain",
    }
)

CONCISE_KEYWORDS: frozenset[str] = frozenset(
    {
        "who",
        "what",
        "when",
        "where",
        "eligible",
        "eligibility",
        "due",
        "deadline",
        "rate",
        "amount",
        "limit",
    }
)


def _token_variants(word: str) -> set[str]:
    variants = {word}
    if len(word) > 1 and word.endswith("s"):
        variants.add(word[:-1])
    return variants


def _tokens(query: str) -> set[str]:
    tokens: set[str] = set()
    for word in re.findall(r"[a-z]+", query.lower()):
        tokens.update(_token_variants(word))
    return tokens


def classify_verbosity(query: str) -> Verbosity:
    """Classify a query as detailed or concise, defaulting to concise.

    Detailed keywords win over concise keywords on a tie (a "how ... when" query is a
    process question and deserves the fuller treatment).
    """
    if _tokens(query) & DETAILED_KEYWORDS:
        return VERBOSITY_DETAILED
    return VERBOSITY_CONCISE


COMPARISON_KEYWORDS: frozenset[str] = frozenset({"compare", "comparison", "difference", "differences", "versus", "vs"})

CALCULATION_KEYWORDS: frozenset[str] = frozenset({"calculate", "calculation", "compute", "computed", "computation"})

PROCEDURAL_KEYWORDS: frozenset[str] = frozenset(
    {"procedure", "process", "step", "steps", "file", "apply", "register", "submit", "obtain"}
)

LEGAL_INTERPRETATION_KEYWORDS: frozenset[str] = frozenset(
    {
        "interpret",
        "interpretation",
        "section",
        "provision",
        "legality",
        "legal",
        "valid",
        "validity",
        "allowed",
        "entitled",
        "permitted",
        "admissible",
        "whether",
    }
)

DEFINITION_KEYWORDS: frozenset[str] = frozenset({"define", "definition", "meaning"})

DEFINITION_DOWNGRADE_KEYWORDS: frozenset[str] = frozenset(
    {"due", "deadline", "rate", "amount", "limit", "date", "year", "who", "when", "where"}
)

DEFINITION_PHRASES: tuple[str, ...] = ("what is", "what are", "whats", "what's")


def _has_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def classify_question_type(query: str) -> QuestionType:
    """Classify a query into a question type, defaulting to FACTUAL.

    Deterministic keyword heuristics with the following precedence:
    COMPARISON > CALCULATION > PROCEDURAL > LEGAL_INTERPRETATION > DEFINITION > FACTUAL.
    """
    text = query.lower()
    tokens = _tokens(query)

    if tokens & COMPARISON_KEYWORDS:
        return QUESTION_TYPE_COMPARISON
    if tokens & CALCULATION_KEYWORDS:
        return QUESTION_TYPE_CALCULATION
    if tokens & PROCEDURAL_KEYWORDS:
        return QUESTION_TYPE_PROCEDURAL
    if tokens & LEGAL_INTERPRETATION_KEYWORDS:
        return QUESTION_TYPE_LEGAL_INTERPRETATION
    is_definition = (tokens & DEFINITION_KEYWORDS) or _has_any_phrase(text, DEFINITION_PHRASES)
    if is_definition and not tokens & DEFINITION_DOWNGRADE_KEYWORDS:
        return QUESTION_TYPE_DEFINITION
    return QUESTION_TYPE_FACTUAL

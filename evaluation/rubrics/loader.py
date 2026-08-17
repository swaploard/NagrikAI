"""Load evaluation rubrics (Markdown) into DeepEval GEval criteria strings."""

from __future__ import annotations

import re
from pathlib import Path

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*.*?```", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def load_geval_criteria(path: str | Path) -> str:
    """Read a rubric markdown file and strip it down to a GEval criteria paragraph.

    Removes JSON code blocks and heading markers, collapsing the grading-scale
    guidance into a single string the judge can follow.
    """
    text = Path(path).read_text(encoding="utf-8")
    text = _JSON_BLOCK_RE.sub(" ", text)
    lines = [re.sub(r"^#+\s*", "", line).strip() for line in text.splitlines()]
    body = " ".join(line for line in lines if line)
    return _WHITESPACE_RE.sub(" ", body).strip()

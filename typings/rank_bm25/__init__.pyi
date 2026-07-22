from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

_T = TypeVar("_T")

class BM25Okapi:
    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        tokenizer: Callable[[str], list[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None: ...
    def get_scores(self, query: Sequence[str]) -> list[float]: ...
    def get_top_n(
        self,
        query: Sequence[str],
        documents: Sequence[_T],
        n: int = 5,
    ) -> list[tuple[float, _T]]: ...

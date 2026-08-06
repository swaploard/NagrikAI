from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, doc["content"]) for doc in documents]
        scores = model.predict(pairs)
        scored = list(zip(scores.tolist(), documents, strict=False))
        scored.sort(key=lambda x: x[0], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        result: list[dict[str, Any]] = []
        for score, doc in scored:
            doc["score"] = float(score)
            result.append(doc)
        return result

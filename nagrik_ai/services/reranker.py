from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import torch

from nagrik_ai.config.config_models import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from nagrik_ai.services.tracing import LangSmithTracer, get_tracer

logger = logging.getLogger(__name__)


class RerankerError(Exception):
    """Base exception for reranker errors."""


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._has_predicted = False

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            load_start = time.perf_counter()
            logger.info("Loading reranker model: %s", self.model_name)
            if torch.cuda.is_available():
                try:
                    self._model = CrossEncoder(self.model_name, torch_dtype=torch.float16)  # type: ignore[call-arg]
                    logger.info("CUDA available: reranker model will run in fp16")
                except TypeError:
                    try:
                        self._model = CrossEncoder(
                            self.model_name,
                            model_kwargs={"torch_dtype": torch.float16},  # type: ignore[call-arg]
                        )
                    except TypeError:
                        self._model = CrossEncoder(
                            self.model_name,
                            automodel_args={"torch_dtype": torch.float16},
                        )
                        logger.info("CUDA available: reranker model will run in fp16 (automodel_args)")
            else:
                self._model = CrossEncoder(self.model_name)
            load_elapsed = (time.perf_counter() - load_start) * 1000
            logger.info("Loaded reranker model: %s (%.0f ms)", self.model_name, load_elapsed)
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
        predict_start = time.perf_counter()
        scores = model.predict(pairs)
        predict_elapsed = (time.perf_counter() - predict_start) * 1000
        if getattr(self, "_has_predicted", False):
            logger.info("Reranker warm predict: %d pairs (%.0f ms)", len(pairs), predict_elapsed)
        else:
            self._has_predicted = True
            logger.info("Reranker first predict (cold): %d pairs (%.0f ms)", len(pairs), predict_elapsed)
        scored = list(zip(scores.tolist(), documents, strict=False))
        scored.sort(key=lambda x: x[0], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        result: list[dict[str, Any]] = []
        for score, doc in scored:
            doc["score"] = float(score)
            result.append(doc)
        return result


class OpenRouterReranker(Reranker):
    """Reranker backed by the OpenRouter rerank API (e.g. cohere/rerank-4-pro)."""

    def __init__(
        self,
        model_name: str = "cohere/rerank-4-pro",
        api_key: str = OPENROUTER_API_KEY,
        base_url: str = OPENROUTER_BASE_URL,
        tracer: LangSmithTracer | None = None,
    ) -> None:
        super().__init__(model_name=model_name)
        if not api_key:
            raise RerankerError(
                "OpenRouter API key is required for reranking. "
                "Set NAGRIKAI_OPENROUTER_API_KEY in .env or environment variables."
            )
        self._api_key = api_key
        self._base_url = str(base_url).rstrip("/")
        self._tracer = tracer or get_tracer()
        self._client = httpx.Client(timeout=60.0)
        logger.info("Initialized OpenRouter reranker with model: %s", model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": [doc["content"] for doc in documents],
        }
        if top_k is not None:
            payload["top_n"] = top_k
        with self._tracer.trace(
            "openrouter_rerank",
            "retriever",
            inputs={"query": query, "model": self.model_name, "documents": len(documents)},
            metadata={"model": self.model_name, "provider": "openrouter", "top_n": top_k},
        ) as span:
            span.start_timer()
            try:
                response = self._client.post(
                    f"{self._base_url}/rerank",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                data: dict[str, Any] = response.json()
            except httpx.HTTPError as e:
                raise RerankerError(f"OpenRouter rerank request failed: {e}") from e
            results: list[dict[str, Any]] = data.get("results", [])
            scored: list[dict[str, Any]] = []
            for item in results:
                doc = dict(documents[item["index"]])
                doc["score"] = float(item["relevance_score"])
                scored.append(doc)
            scored.sort(key=lambda x: x["score"], reverse=True)
            outputs: dict[str, Any] = {
                "documents": len(scored),
                "latency_ms": span.elapsed_ms(),
            }
            if isinstance(data.get("usage"), dict):
                outputs["usage"] = data["usage"]
            span.set_outputs(outputs)
            return scored

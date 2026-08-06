from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from nagrik_ai.config.config_models import NagrikAIConfig


class ConfigManager:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or Path(__file__).parent / "site_configs.yaml"
        self._config: NagrikAIConfig | None = None
        load_dotenv()

    def load(self) -> NagrikAIConfig:
        if self._config is not None:
            return self._config

        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        raw = self._apply_env_overrides(raw)
        try:
            self._config = NagrikAIConfig(**raw)
        except ValidationError as e:
            msg = f"Invalid configuration in {self.config_path}"
            raise ValueError(msg) from e
        return self._config

    def _apply_env_overrides(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not raw:
            raw = {}

        env_mappings: dict[str, str] = {
            "chroma_persist_dir": "NAGRIKAI_CHROMA_PERSIST_DIR",
            "content_dir": "NAGRIKAI_CONTENT_DIR",
            "embedding_model": "NAGRIKAI_EMBEDDING_MODEL",
            "llm_provider": "NAGRIKAI_LLM_PROVIDER",
            "ollama_base_url": "NAGRIKAI_OLLAMA_BASE_URL",
            "ollama_model": "NAGRIKAI_OLLAMA_MODEL",
            "chunk_size": "NAGRIKAI_CHUNK_SIZE",
            "chunk_overlap": "NAGRIKAI_CHUNK_OVERLAP",
            "top_k": "NAGRIKAI_TOP_K",
            "fetch_k": "NAGRIKAI_FETCH_K",
            "lambda_mult": "NAGRIKAI_LAMBDA_MULT",
            "reranker_model": "NAGRIKAI_RERANKER_MODEL",
            "reranker_enabled": "NAGRIKAI_RERANKER_ENABLED",
            "hybrid_search_enabled": "NAGRIKAI_HYBRID_SEARCH_ENABLED",
            "bm25_k1": "NAGRIKAI_BM25_K1",
            "bm25_b": "NAGRIKAI_BM25_B",
            "rrf_k": "NAGRIKAI_RRF_K",
            "authority_ranking_enabled": "NAGRIKAI_AUTHORITY_RANKING_ENABLED",
            "max_response_tokens": "NAGRIKAI_MAX_RESPONSE_TOKENS",
        }

        for key, env_var in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                raw[key] = self._convert_value(key, value)

        openrouter_env_mappings: dict[str, str] = {
            "api_key": "NAGRIKAI_OPENROUTER_API_KEY",
            "base_url": "NAGRIKAI_OPENROUTER_BASE_URL",
            "model": "NAGRIKAI_OPENROUTER_MODEL",
        }

        if "openrouter" not in raw:
            raw["openrouter"] = {}
        for key, env_var in openrouter_env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                raw["openrouter"][key] = value

        return raw

    def _convert_value(self, key: str, value: str) -> str | int | float | bool:
        if key in {"chunk_size", "chunk_overlap", "top_k", "fetch_k", "rrf_k", "max_response_tokens"}:
            return int(value)
        if key in {"lambda_mult", "bm25_k1", "bm25_b"}:
            return float(value)
        if key in {"reranker_enabled", "hybrid_search_enabled", "authority_ranking_enabled"}:
            return value.lower() in {"true", "1", "yes", "on"}
        return value

    @property
    def config(self) -> NagrikAIConfig:
        if self._config is None:
            return self.load()
        return self._config

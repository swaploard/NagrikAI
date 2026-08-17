"""DeepEval judge model factory backed by NAGRIKAI_JUDGE_LLM_* environment settings."""

from __future__ import annotations

from deepeval.models import DeepEvalBaseLLM, OllamaModel, OpenRouterModel

from nagrik_ai.config.config_models import (
    JUDGE_LLM_API_KEY,
    JUDGE_LLM_BASE_URL,
    JUDGE_LLM_MODEL,
    JUDGE_LLM_PROVIDER,
    JUDGE_LLM_TEMPERATURE,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


class JudgeConfigError(RuntimeError):
    """Raised when the judge LLM configuration is invalid."""


def create_eval_judge(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
) -> DeepEvalBaseLLM:
    """Build a DeepEval judge model.

    Provider resolution order: explicit argument, NAGRIKAI_JUDGE_LLM_PROVIDER,
    then the app-level NAGRIKAI_LLM_PROVIDER. Model and connection details fall
    back to NAGRIKAI_JUDGE_LLM_* settings and finally to provider defaults.
    """
    resolved_provider = provider or JUDGE_LLM_PROVIDER or LLM_PROVIDER
    resolved_temperature = JUDGE_LLM_TEMPERATURE if temperature is None else temperature
    if resolved_provider == "ollama":
        return OllamaModel(
            model=model or JUDGE_LLM_MODEL or "qwen2.5:7b",
            base_url=base_url or JUDGE_LLM_BASE_URL or OLLAMA_BASE_URL,
            temperature=resolved_temperature,
        )
    if resolved_provider == "openrouter":
        return OpenRouterModel(
            model=model or JUDGE_LLM_MODEL or "openrouter/auto",
            api_key=api_key or JUDGE_LLM_API_KEY or OPENROUTER_API_KEY,
            base_url=base_url or JUDGE_LLM_BASE_URL or OPENROUTER_BASE_URL,
            temperature=resolved_temperature,
        )
    raise JudgeConfigError(f"Unsupported judge provider: {resolved_provider!r}")

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator

from ollama import Client as OllamaClient
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from nagrik_ai.config.config_models import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM service errors."""
    pass


class AuthenticationError(LLMError):
    """Raised when authentication fails."""
    pass


class BaseLLMService(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        pass


class OllamaLLMService(BaseLLMService):
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL) -> None:
        self.client = OllamaClient(host=base_url)
        self.model = model
        logger.info("Initialized Ollama LLM service with model: %s", model)

    def generate(self, prompt: str, system: str | None = None) -> str:
        options: dict[str, str] = {}
        if system:
            options["system"] = system
        response = self.client.generate(model=self.model, prompt=prompt, options=options)
        return response.response or ""

    def generate_stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        options: dict[str, str] = {}
        if system:
            options["system"] = system
        stream = self.client.generate(model=self.model, prompt=prompt, options=options, stream=True)
        for chunk in stream:
            yield chunk.response or ""


class OpenRouterLLMService(BaseLLMService):
    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        base_url: str = OPENROUTER_BASE_URL,
        model: str = OPENROUTER_MODEL,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        logger.info("Initialized OpenRouter LLM service with model: %s", model)

    def _handle_error(self, e: Exception) -> None:
        error_msg = str(e).lower()
        if "401" in error_msg or "unauthorized" in error_msg or "auth" in error_msg:
            raise AuthenticationError(
                "OpenRouter authentication failed. "
                "Please check your NAGRIKAI_OPENROUTER_API_KEY in .env or environment variables. "
                "Get a key from https://openrouter.ai/keys"
            ) from e
        raise LLMError(f"OpenRouter API error: {e}") from e

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            self._handle_error(e)

    def generate_stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self._handle_error(e)


def create_llm_service(
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> BaseLLMService:
    provider = provider or LLM_PROVIDER
    if provider == "openrouter":
        return OpenRouterLLMService(
            api_key=api_key or OPENROUTER_API_KEY,
            base_url=base_url or OPENROUTER_BASE_URL,
            model=model or OPENROUTER_MODEL,
        )
    return OllamaLLMService(
        base_url=base_url or OLLAMA_BASE_URL,
        model=model or OLLAMA_MODEL,
    )


LLMService = create_llm_service

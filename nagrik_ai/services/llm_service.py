from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, NoReturn, cast

from ollama import ChatResponse, Message
from ollama import Client as OllamaClient
from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
)

from nagrik_ai.config.config_models import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)
from nagrik_ai.services.tracing import LangSmithTracer, get_tracer

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM service errors."""

    pass


class AuthenticationError(LLMError):
    """Raised when authentication fails."""

    pass


class RateLimitError(LLMError):
    """Raised when the upstream provider rejects the request due to rate limits or exhausted capacity."""

    pass


@dataclass
class ToolCall:
    """A tool call returned by the LLM."""

    name: str = ""
    arguments: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    id: str = ""


@dataclass
class LLMResponse:
    """A structured response from the LLM, possibly containing tool calls."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class BaseLLMService(ABC):
    def __init__(
        self,
        tracer: LangSmithTracer | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        self.tracer: LangSmithTracer = tracer or get_tracer()
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Send messages and optional tool definitions, returning a structured response."""
        pass


class OllamaLLMService(BaseLLMService):
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        tracer: LangSmithTracer | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(tracer=tracer, temperature=temperature, max_tokens=max_tokens)
        self.client = OllamaClient(host=base_url)
        self.model = model
        logger.info("Initialized Ollama LLM service with model: %s", model)

    def _build_options(self, system: str | None = None) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if system:
            options["system"] = system
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        return options

    def generate(self, prompt: str, system: str | None = None) -> str:
        options = self._build_options(system)
        logger.info("About to call Ollama generate (model: %s)", self.model)
        with self.tracer.trace(
            "ollama_generate",
            "llm",
            inputs={"prompt": prompt, "model": self.model, "system": system},
            metadata={
                "model": self.model,
                "provider": "ollama",
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        ) as span:
            span.start_timer()
            response = self.client.generate(model=self.model, prompt=prompt, options=options)
            result = response.response or ""
            latency = span.elapsed_ms()
            outputs: dict[str, Any] = {
                "response_length": len(result),
                "latency_ms": latency,
            }
            token_usage: dict[str, int] = {}
            if response.prompt_eval_count is not None:
                token_usage["input"] = response.prompt_eval_count
            if response.eval_count is not None:
                token_usage["output"] = response.eval_count
            if token_usage:
                if response.prompt_eval_count is not None and response.eval_count is not None:
                    token_usage["total"] = response.prompt_eval_count + response.eval_count
                outputs["token_usage"] = token_usage
            if response.done_reason:
                outputs["finish_reason"] = response.done_reason
            if response.total_duration:
                duration_ms = response.total_duration / 1_000_000
                outputs["total_duration_ms"] = duration_ms
            span.set_outputs(outputs)
            return result

    def generate_stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        options = self._build_options(system)
        logger.info("About to call Ollama generate with stream (model: %s)", self.model)
        with self.tracer.trace(
            "ollama_generate_stream",
            "llm",
            inputs={"prompt": prompt, "model": self.model, "system": system},
            metadata={
                "model": self.model,
                "provider": "ollama",
                "streaming": True,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        ) as span:
            span.start_timer()
            stream = self.client.generate(model=self.model, prompt=prompt, options=options, stream=True)
            full_response = ""
            final_meta: dict[str, Any] = {}
            for chunk in stream:
                chunk_text = chunk.response or ""
                full_response += chunk_text
                if chunk.done:
                    final_meta["prompt_eval_count"] = chunk.prompt_eval_count
                    final_meta["eval_count"] = chunk.eval_count
                    final_meta["done_reason"] = chunk.done_reason
                    final_meta["total_duration"] = chunk.total_duration
                yield chunk_text
            latency = span.elapsed_ms()
            outputs: dict[str, Any] = {
                "response_length": len(full_response),
                "latency_ms": latency,
            }
            token_usage = {}
            if final_meta.get("prompt_eval_count") is not None:
                token_usage["input"] = final_meta["prompt_eval_count"]
            if final_meta.get("eval_count") is not None:
                token_usage["output"] = final_meta["eval_count"]
            if token_usage:
                inp = final_meta.get("prompt_eval_count")
                out = final_meta.get("eval_count")
                if inp is not None and out is not None:
                    token_usage["total"] = inp + out
                outputs["token_usage"] = token_usage
            if final_meta.get("done_reason"):
                outputs["finish_reason"] = final_meta["done_reason"]
            td = final_meta.get("total_duration")
            if td:
                outputs["total_duration_ms"] = td / 1_000_000
            span.set_outputs(outputs)


    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        chat_messages = list(messages)
        if system:
            chat_messages.insert(0, {"role": "system", "content": system})

        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "options": options,
        }
        if tools:
            kwargs["tools"] = tools

        logger.info("Ollama chat (model: %s, tools: %d)", self.model, len(tools or []))

        with self.tracer.trace(
            "ollama_chat",
            "llm",
            inputs={"messages": messages, "model": self.model, "system": system},
            metadata={
                "model": self.model,
                "provider": "ollama",
                "num_tools": len(tools or []),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        ) as span:
            span.start_timer()
            response: ChatResponse = self.client.chat(**kwargs)  # type: ignore[reportUnknownMemberType]
            message: Message = cast(Message, response.message)  # type: ignore[reportUnknownMemberType]
            latency = span.elapsed_ms()

            tool_calls: list[ToolCall] | None = None
            if message.tool_calls:
                tool_calls = []
                for idx, tc in enumerate(message.tool_calls):
                    args = tc.function.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                    if not isinstance(args, dict):
                        args = dict(args)
                    parsed_args = cast("dict[str, Any]", args)
                    tool_calls.append(ToolCall(
                        name=tc.function.name,
                        arguments=parsed_args,
                        id=getattr(tc, "id", "") or f"call_{idx}",
                    ))

            outputs: dict[str, Any] = {
                "latency_ms": latency,
                "has_tool_calls": tool_calls is not None,
            }
            if message.content is not None:
                outputs["response_length"] = len(message.content)
            span.set_outputs(outputs)

            return LLMResponse(content=message.content, tool_calls=tool_calls)


class OpenRouterLLMService(BaseLLMService):
    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        base_url: str = OPENROUTER_BASE_URL,
        model: str = OPENROUTER_MODEL,
        tracer: LangSmithTracer | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(tracer=tracer, temperature=temperature, max_tokens=max_tokens)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        logger.info("Initialized OpenRouter LLM service with model: %s", model)

    def _handle_error(self, e: Exception) -> NoReturn:
        error_msg = str(e).lower()
        if "401" in error_msg or "unauthorized" in error_msg or "auth" in error_msg:
            raise AuthenticationError(
                "OpenRouter authentication failed. "
                "Please check your NAGRIKAI_OPENROUTER_API_KEY in .env or environment variables. "
                "Get a key from https://openrouter.ai/keys"
            ) from e
        if any(
            token in error_msg
            for token in (
                "rate limit",
                "ratelimit",
                "too many requests",
                "429",
                "resource exhausted",
                "resourceexhausted",
                "capacity",
                "quota",
                "worker local total request limit",
            )
        ):
            raise RateLimitError(
                "The AI provider is currently at capacity or rate-limited. "
                "Please wait a moment and try again. If this persists, consider switching models "
                "or checking your OpenRouter plan limits."
            ) from e
        raise LLMError(f"OpenRouter API error: {e}") from e

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        logger.info("About to call OpenRouter chat.completions (model: %s)", self.model)
        try:
            with self.tracer.trace(
                "openrouter_generate",
                "llm",
                inputs={"prompt": prompt, "model": self.model, "system": system},
                metadata={
                    "model": self.model,
                    "provider": "openrouter",
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            ) as span:
                span.start_timer()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                result = response.choices[0].message.content or ""
                latency = span.elapsed_ms()
                outputs: dict[str, Any] = {
                    "response_length": len(result),
                    "latency_ms": latency,
                }
                if response.usage:
                    usage = response.usage
                    token_usage: dict[str, int] = {
                        "input": usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total": usage.total_tokens,
                    }
                    if token_usage:
                        outputs["token_usage"] = token_usage
                if response.choices and response.choices[0].finish_reason:
                    outputs["finish_reason"] = response.choices[0].finish_reason
                span.set_outputs(outputs)
                return result
        except Exception as e:
            self._handle_error(e)

    def generate_stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        span = None
        logger.info("About to call OpenRouter chat.completions with streaming (model: %s)", self.model)
        try:
            with self.tracer.trace(
                "openrouter_generate_stream",
                "llm",
                inputs={"prompt": prompt, "model": self.model, "system": system},
                metadata={
                    "model": self.model,
                    "provider": "openrouter",
                    "streaming": True,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            ) as span:
                span.start_timer()
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                logger.info("Starting to iterate over OpenRouter stream")
                full_response = ""
                finish_reason: str | None = None
                usage_data: dict[str, int] = {}
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunk_text = chunk.choices[0].delta.content
                        full_response += chunk_text
                        yield chunk_text
                    if chunk.choices and chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                    if chunk.usage:
                        usage_data["input"] = chunk.usage.prompt_tokens
                        usage_data["output"] = chunk.usage.completion_tokens
                        usage_data["total"] = chunk.usage.total_tokens
                latency = span.elapsed_ms()
                outputs: dict[str, Any] = {
                    "response_length": len(full_response),
                    "latency_ms": latency,
                }
                if usage_data:
                    outputs["token_usage"] = usage_data
                if finish_reason:
                    outputs["finish_reason"] = finish_reason
                span.set_outputs(outputs)
        except Exception as e:
            if span is not None:
                span.set_outputs({"error": str(e)})
            self._handle_error(e)


    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        chat_messages: list[ChatCompletionMessageParam] = list(messages)  # type: ignore[arg-type]
        if system and not any(m.get("role") == "system" for m in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": system})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        logger.info("OpenRouter chat (model: %s, tools: %d)", self.model, len(tools or []))

        try:
            with self.tracer.trace(
                "openrouter_chat",
                "llm",
                inputs={"messages": messages, "model": self.model, "system": system},
                metadata={
                    "model": self.model,
                    "provider": "openrouter",
                    "num_tools": len(tools or []),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            ) as span:
                span.start_timer()
                response = cast(
                    ChatCompletion, self.client.chat.completions.create(**kwargs)
                )
                choice = response.choices[0]
                message = choice.message
                latency = span.elapsed_ms()

                tool_calls: list[ToolCall] | None = None
                if message.tool_calls:
                    parsed_tool_calls: list[ToolCall] = []
                    for idx, tc in enumerate(message.tool_calls):
                        if not isinstance(tc, ChatCompletionMessageToolCall):
                            continue
                        name = tc.function.name
                        arguments = json.loads(tc.function.arguments)
                        parsed_tool_calls.append(
                            ToolCall(
                                name=name,
                                arguments=arguments,
                                id=tc.id or f"call_{idx}",
                            )
                        )
                    tool_calls = parsed_tool_calls

                outputs: dict[str, Any] = {
                    "latency_ms": latency,
                    "has_tool_calls": tool_calls is not None,
                }
                if message.content is not None:
                    outputs["response_length"] = len(message.content)
                if response.usage:
                    usage = response.usage
                    token_usage: dict[str, int] = {
                        "input": usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total": usage.total_tokens,
                    }
                    outputs["token_usage"] = token_usage
                span.set_outputs(outputs)

                return LLMResponse(content=message.content, tool_calls=tool_calls)
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

"""Shared timing, stable evidence IDs and public failure semantics for tools."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import replace
from functools import wraps
from hashlib import sha256

import httpx
import requests

from nagrik_ai.models.tool_result import ErrorCode, ToolResult

logger = logging.getLogger(__name__)


def source_id(prefix: str, value: str | bytes) -> str:
    payload = value.encode() if isinstance(value, str) else value
    return f"{prefix}:{sha256(payload).hexdigest()[:24]}"


def failure(message: str, code: ErrorCode = "VALIDATION_ERROR") -> ToolResult:
    return ToolResult(False, None, code, message, None, None)


def tool_result[**P](fn: Callable[P, ToolResult]) -> Callable[P, ToolResult]:
    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> ToolResult:
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except (TimeoutError, httpx.TimeoutException, requests.exceptions.Timeout):  # fmt: skip
            result = failure("Tool request timed out.", "TIMEOUT")
        except FileNotFoundError:
            result = failure("Requested evidence was not found.", "NOT_FOUND")
        except Exception as exc:
            from nagrik_ai.services.llm_service import RateLimitError

            cause = exc.__cause__ or exc
            status = getattr(getattr(cause, "response", None), "status_code", None)
            code: ErrorCode = "VALIDATION_ERROR"
            if isinstance(exc, RateLimitError) or status == 429:
                code = "RATE_LIMITED"
            elif isinstance(cause, (TimeoutError, httpx.TimeoutException, requests.exceptions.Timeout)):
                code = "TIMEOUT"
            result = failure(f"Tool failed ({type(exc).__name__}).", code)
        result = replace(result, latency_ms=(time.perf_counter() - start) * 1000)
        logger.info(
            "Tool %s: ok=%s code=%s latency_ms=%.2f", fn.__name__, result.ok, result.error_code, result.latency_ms
        )
        return result

    return wrapped

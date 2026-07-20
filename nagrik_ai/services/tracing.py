from __future__ import annotations

import datetime
import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal
from uuid import UUID, uuid4

from langsmith import Client

from nagrik_ai.config.config_models import (
    LANGSMITH_API_KEY,
    LANGSMITH_ENDPOINT,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACE_VERBOSE,
    LANGSMITH_TRACING_ENABLED,
)

logger = logging.getLogger(__name__)

RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]

_current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)


class LangSmithSpan:
    def __init__(self, session_id: str | None = None, user_id: str | None = None) -> None:
        self.outputs: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self._start_time: float | None = None
        self.session_id = session_id
        self.user_id = user_id

    def set_outputs(self, outputs: dict[str, Any]) -> None:
        self.outputs.update(outputs)

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata.update(metadata)

    def start_timer(self) -> None:
        self._start_time = time.perf_counter()

    def elapsed_ms(self) -> float:
        if self._start_time is None:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000


class LangSmithTracer:
    def __init__(
        self,
        project_name: str = LANGSMITH_PROJECT,
        api_key: str | None = LANGSMITH_API_KEY,
        endpoint: str = LANGSMITH_ENDPOINT,
        enabled: bool = LANGSMITH_TRACING_ENABLED,
        verbose: bool = LANGSMITH_TRACE_VERBOSE,
    ) -> None:
        self._client: Client | None = None
        self._enabled = enabled
        self._project_name = project_name
        self._verbose = verbose

        if not enabled:
            return

        if not api_key:
            logger.warning("LangSmith tracing enabled but LANGSMITH_API_KEY not set")
            self._enabled = False
            return

        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", project_name)
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
        os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint)

        try:
            self._client = Client(api_url=endpoint, api_key=api_key)
            logger.info(
                "LangSmith tracing initialized (project: %s, endpoint: %s)",
                project_name,
                endpoint,
            )
        except Exception:
            logger.exception("Failed to initialize LangSmith client")
            self._enabled = False

    @contextmanager
    def trace(
        self,
        name: str,
        run_type: RunType,
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> Generator[LangSmithSpan]:
        if not self._enabled or self._client is None:
            span = LangSmithSpan(session_id=session_id, user_id=user_id)
            yield span
            return

        run_id: UUID = uuid4()
        parent_run_id: str | None = _current_run_id.get()
        start_time = datetime.datetime.now(datetime.UTC)

        inputs = dict(inputs or {})
        if session_id:
            inputs.setdefault("session_id", session_id)
        if user_id:
            inputs.setdefault("user_id", user_id)

        extra: dict[str, Any] = {}
        if metadata:
            extra["metadata"] = metadata

        create_kwargs: dict[str, Any] = {
            "id": run_id,
            "start_time": start_time,
        }
        if tags:
            create_kwargs["tags"] = tags
        if extra:
            create_kwargs["extra"] = extra
        if parent_run_id:
            create_kwargs["parent_run_id"] = parent_run_id

        self._client.create_run(
            name=name,
            inputs=inputs or {},
            run_type=run_type,
            project_name=self._project_name,
            **create_kwargs,
        )

        span = LangSmithSpan(session_id=session_id, user_id=user_id)
        if metadata:
            span.set_metadata(metadata)

        _current_run_id.set(str(run_id))
        try:
            yield span
        except GeneratorExit:
            span.metadata["early_exit"] = True
            update_kwargs: dict[str, Any] = {"end_time": datetime.datetime.now(datetime.UTC)}
            if span.outputs:
                update_kwargs["outputs"] = dict(span.outputs)
            if span.metadata:
                update_kwargs["extra"] = {"metadata": dict(span.metadata)}
            self._client.update_run(run_id=run_id, **update_kwargs)  # type: ignore[arg-type]
            raise
        except Exception as e:
            update_kwargs: dict[str, Any] = {
                "error": f"{type(e).__name__}: {e}",
                "end_time": datetime.datetime.now(datetime.UTC),
            }
            if span.outputs:
                update_kwargs["outputs"] = dict(span.outputs)
            self._client.update_run(run_id=run_id, **update_kwargs)  # type: ignore[arg-type]
            raise
        else:
            update_kwargs: dict[str, Any] = {"end_time": datetime.datetime.now(datetime.UTC)}
            if span.outputs:
                update_kwargs["outputs"] = dict(span.outputs)
            if span.metadata:
                extra_meta = {"metadata": dict(span.metadata)}
                update_kwargs["extra"] = extra_meta
            self._client.update_run(run_id=run_id, **update_kwargs)  # type: ignore[arg-type]
        finally:
            _current_run_id.set(parent_run_id)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def client(self) -> Client | None:
        return self._client

    @property
    def verbose(self) -> bool:
        return self._verbose


_tracer: LangSmithTracer | None = None


def get_tracer() -> LangSmithTracer:
    global _tracer
    if _tracer is None:
        _tracer = LangSmithTracer()
    return _tracer


def configure_tracing(
    project_name: str = LANGSMITH_PROJECT,
    api_key: str = LANGSMITH_API_KEY,
    endpoint: str = LANGSMITH_ENDPOINT,
    enabled: bool = LANGSMITH_TRACING_ENABLED,
    verbose: bool = LANGSMITH_TRACE_VERBOSE,
) -> LangSmithTracer:
    global _tracer
    _tracer = LangSmithTracer(
        project_name=project_name,
        api_key=api_key,
        endpoint=endpoint,
        enabled=enabled,
        verbose=verbose,
    )
    return _tracer

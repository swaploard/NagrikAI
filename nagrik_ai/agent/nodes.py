from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any

from nagrik_ai.agent.validation import CONCISE_RESPONSE_WORD_CAP, ValidationStatus, validate_response
from nagrik_ai.agent.verbosity import (
    QUESTION_TYPE_FACTUAL,
    VERBOSITY_CONCISE,
    VERBOSITY_DETAILED,
    classify_question_type,
    classify_verbosity,
)
from nagrik_ai.config.config_models import (
    MAX_RESPONSE_TOKENS,
    MAX_RESPONSE_TOKENS_DETAILED,
    MAX_RESPONSE_TOKENS_HARD,
)
from nagrik_ai.models.agent_state import AgentState
from nagrik_ai.models.rag_result import RAGResult, SourceInfo
from nagrik_ai.prompts.prompt_loader import get_prompt_version, load_prompt
from nagrik_ai.prompts.prompt_registry import CompiledPromptPipeline, load_default_prompt_pipeline
from nagrik_ai.services.citation_service import (
    citation_sort_key,
    extract_snippet,
    flatten_doc,
    format_merged_context_block,
    source_group_key,
)
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.llm_service import BaseLLMService
from nagrik_ai.services.reranker import Reranker
from nagrik_ai.services.tracing import LangSmithTracer, get_tracer

logger = logging.getLogger(__name__)

NO_RETRIEVAL_RESPONSE = "No authoritative source was found in the retrieved documents."


def response_token_budget(metadata: dict[str, Any]) -> int:
    """Token budget for the next generation, scaled by verbosity and retry count.

    Concise queries get the base budget, detailed queries get a larger one, and every
    retry doubles the budget (bounded by the hard ceiling set in configuration).
    """
    verbosity = str(metadata.get("verbosity", VERBOSITY_CONCISE))
    base = int(MAX_RESPONSE_TOKENS_DETAILED if verbosity == VERBOSITY_DETAILED else MAX_RESPONSE_TOKENS)
    raw_retry = metadata.get("retry_count")
    retries = int(raw_retry) if isinstance(raw_retry, (int, float, str)) else 0
    ceiling = int(MAX_RESPONSE_TOKENS_HARD)
    return min(int(base * (2**retries)), ceiling)


def classify_node(
    state: AgentState,
    tracer: LangSmithTracer | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Deterministically classify the query's verbosity and question type into metadata."""
    _tracer = tracer or get_tracer()
    query = state.get("query", "")
    verbosity = classify_verbosity(query)
    question_type = classify_question_type(query)
    with _tracer.trace(
        "classify_query",
        "chain",
        inputs={"query": query},
        session_id=session_id,
        user_id=user_id,
    ) as span:
        span.set_outputs({"verbosity": verbosity, "question_type": question_type})
    return {
        "metadata": {
            **state.get("metadata", {}),
            "verbosity": verbosity,
            "question_type": question_type,
        }
    }


def retrieve_node(
    state: AgentState,
    retrieval_service: DocumentRetrievalService,
    tracer: LangSmithTracer | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    _tracer = tracer or get_tracer()
    query = state["query"]
    start_time = time.perf_counter()
    with _tracer.trace(
        "retrieve_documents", "retriever", inputs={"query": query}, session_id=session_id, user_id=user_id
    ) as span:
        docs = retrieval_service.retrieve(query)
        span.set_outputs({"documents": len(docs)})
        logger.info("Retrieved %d documents", len(docs))
    return {
        "documents": docs,
        "metadata": {**state.get("metadata", {}), "retrieved_count": len(docs), "start_time": start_time},
    }


def rerank_node(
    state: AgentState,
    reranker: Reranker | None = None,
    tracer: LangSmithTracer | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    if reranker is None:
        return {}
    _tracer = tracer or get_tracer()
    docs = state["documents"]
    with _tracer.trace(
        "rerank_documents",
        "retriever",
        inputs={"query": state["query"], "documents": len(docs)},
        session_id=session_id,
        user_id=user_id,
    ) as span:
        reranked_docs = reranker.rerank(state["query"], docs)
        span.set_outputs({"documents": len(reranked_docs)})
    return {"documents": reranked_docs}


def build_context_node(
    state: AgentState,
    tracer: LangSmithTracer | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    _tracer = tracer or get_tracer()
    query = state["query"]
    results = state["documents"]

    with _tracer.trace(
        "build_context",
        "chain",
        inputs={"query": query, "documents": len(results)},
        session_id=session_id,
        user_id=user_id,
    ) as span:
        sorted_results = sorted(results, key=citation_sort_key)

        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for doc in sorted_results:
            group_key = source_group_key(flatten_doc(doc))
            grouped.setdefault(group_key, []).append(doc)

        display_sources: list[SourceInfo] = []
        context_blocks: list[str] = []
        locked_results: list[dict[str, Any]] = []
        for i, (group_key, group_docs) in enumerate(grouped.items(), start=1):
            first_flat = flatten_doc(group_docs[0])
            display_sources.append(
                SourceInfo(
                    title=str(first_flat.get("title", "Unknown")),
                    url=str(first_flat.get("url", "")),
                    domain=str(first_flat.get("domain", "")),
                    source_id=group_key[0],
                    citation_id=i,
                    chunk_index=int(first_flat.get("chunk_index", 0)),
                    total_chunks=int(first_flat.get("total_chunks", 1)),
                    score=float(first_flat.get("score", 0.0)),
                    snippet=extract_snippet(
                        str(first_flat.get("page_content", first_flat.get("content", ""))),
                        query,
                    ),
                )
            )
            context_blocks.append(format_merged_context_block([flatten_doc(d) for d in group_docs], i))
            for doc in group_docs:
                doc["citation_id"] = i
                locked_results.append(doc)

        context = "\n\n---\n\n".join(context_blocks)
        retrieval_fingerprint = {
            "context_hash": hashlib.sha256(context.encode("utf-8")).hexdigest()[:12],
            "retriever": state.get("retrieval_config", {}).get("retriever", "configured"),
            "top_k": state.get("retrieval_config", {}).get("top_k"),
            "filters": state.get("retrieval_config", {}).get("filters", {}),
            "documents": [
                {
                    "source_id": src.source_id,
                    "citation_id": src.citation_id,
                    "chunk_index": src.chunk_index,
                    "score": src.score,
                }
                for src in display_sources
            ],
        }

        span.set_outputs({"context_length": len(context), "sources": len(display_sources)})

    return {
        "context": context,
        "citations": [
            {
                "citation_id": src.citation_id,
                "source_id": src.source_id,
                "title": src.title,
                "url": src.url,
                "domain": src.domain,
                "score": src.score,
                "snippet": src.snippet,
                "chunk_index": src.chunk_index,
                "total_chunks": src.total_chunks,
            }
            for src in display_sources
        ],
        "documents": locked_results,
        "metadata": {
            **state.get("metadata", {}),
            "retrieval_fingerprint": retrieval_fingerprint,
        },
    }


def _render_generation_prompt(
    state: AgentState,
    *,
    system_prompt: str,
    compiled_pipeline: CompiledPromptPipeline | None,
) -> tuple[str, dict[str, Any]]:
    context = state.get("context", "") or ""
    metadata = state.get("metadata", {})
    verbosity = metadata.get("verbosity", VERBOSITY_CONCISE)
    question_type = metadata.get("question_type", QUESTION_TYPE_FACTUAL)

    if system_prompt:
        user_prompt = load_prompt(
            "user_query",
            question=state["query"],
            verbosity=verbosity,
            question_type=question_type,
        )
        combined = (
            f"{system_prompt}\n\n## Context\n{context}\n\n{user_prompt}"
            if context
            else f"{system_prompt}\n\n{user_prompt}"
        )
        return combined, {
            "prompt_versions": {
                "system_prompt": get_prompt_version("system_prompt"),
                "user_query": get_prompt_version("user_query"),
            }
        }

    pipeline = compiled_pipeline or load_default_prompt_pipeline()
    rendered = pipeline.render(
        context=context,
        question=state["query"],
        verbosity=str(verbosity),
        question_type=str(question_type),
    )
    return rendered.text, {
        "pipeline_id": rendered.pipeline_id,
        "pipeline_version": rendered.pipeline_version,
        "prompt_content_hash": rendered.prompt_content_hash,
    }


def generate_node(
    state: AgentState,
    llm_service: BaseLLMService | None,
    system_prompt: str = "",
    compiled_pipeline: CompiledPromptPipeline | None = None,
    tracer: LangSmithTracer | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    if llm_service is None:
        error_msg = "llm_service is required for generate_node"
        return {"answer": error_msg, "errors": [error_msg], "candidate_answers": []}

    _tracer = tracer or get_tracer()
    metadata = state.get("metadata", {})
    if metadata.get("retrieved_count") == 0:
        return {
            "answer": NO_RETRIEVAL_RESPONSE,
            "candidate_answers": [NO_RETRIEVAL_RESPONSE],
            "metadata": {
                **dict(metadata),
                "no_answer_reason": "no_retrieval",
                "validation_status": ValidationStatus.FAIL.value,
                "validation_retryable": False,
            },
        }

    combined, prompt_metadata = _render_generation_prompt(
        state,
        system_prompt=system_prompt,
        compiled_pipeline=compiled_pipeline,
    )
    max_tokens = response_token_budget(metadata)

    with _tracer.trace(
        "format_prompt",
        "prompt",
        inputs={"prompt": combined, "question": state["query"]},
        metadata={
            **prompt_metadata,
            "retrieval_fingerprint": metadata.get("retrieval_fingerprint", {}),
            "verbosity": metadata.get("verbosity", VERBOSITY_CONCISE),
            "question_type": metadata.get("question_type", QUESTION_TYPE_FACTUAL),
            "max_tokens": max_tokens,
        },
        session_id=session_id,
        user_id=user_id,
    ):
        pass

    response = llm_service.generate(combined, system=None, max_tokens=max_tokens)
    finish_reason = getattr(llm_service, "last_finish_reason", None)
    logger.info("generate_node: LLM response=%s... (finish_reason=%s)", response[:200], finish_reason)
    return {
        "answer": response,
        "candidate_answers": [response],
        "metadata": {
            **dict(state.get("metadata", {})),
            **prompt_metadata,
            "finish_reason": finish_reason or "",
            "max_tokens": max_tokens,
        },
    }


def generate_stream_node(
    state: AgentState,
    llm_service: BaseLLMService | None,
    system_prompt: str = "",
    compiled_pipeline: CompiledPromptPipeline | None = None,
    tracer: LangSmithTracer | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Streaming version of generate_node - uses callback for token streaming."""
    if llm_service is None:
        error_msg = "llm_service is required for generate_stream_node"
        return {"answer": error_msg, "errors": [error_msg], "candidate_answers": []}

    _tracer = tracer or get_tracer()
    metadata = state.get("metadata", {})
    streaming_buffer = state.get("_streaming_buffer") or []
    streaming_callback = state.get("_streaming_callback")

    if metadata.get("retrieved_count") == 0:
        for token in NO_RETRIEVAL_RESPONSE.split(" "):
            chunk = token + " "
            streaming_buffer.append(chunk)
            if streaming_callback is not None:
                streaming_callback(chunk)
        return {
            "answer": NO_RETRIEVAL_RESPONSE,
            "candidate_answers": [NO_RETRIEVAL_RESPONSE],
            "_streaming_buffer": streaming_buffer,
            "metadata": {
                **dict(metadata),
                "no_answer_reason": "no_retrieval",
                "validation_status": ValidationStatus.FAIL.value,
                "validation_retryable": False,
            },
        }

    combined, prompt_metadata = _render_generation_prompt(
        state,
        system_prompt=system_prompt,
        compiled_pipeline=compiled_pipeline,
    )
    max_tokens = response_token_budget(metadata)

    with _tracer.trace(
        "format_prompt",
        "prompt",
        inputs={"prompt": combined, "question": state["query"]},
        metadata={
            **prompt_metadata,
            "retrieval_fingerprint": metadata.get("retrieval_fingerprint", {}),
            "verbosity": metadata.get("verbosity", VERBOSITY_CONCISE),
            "question_type": metadata.get("question_type", QUESTION_TYPE_FACTUAL),
            "max_tokens": max_tokens,
        },
        session_id=session_id,
        user_id=user_id,
    ):
        pass

    full_response = ""

    with _tracer.trace(
        "generate_stream", "llm", inputs={"query": state["query"]}, session_id=session_id, user_id=user_id
    ) as span:
        for chunk in llm_service.generate_stream(combined, system=None, max_tokens=max_tokens):
            full_response += chunk
            streaming_buffer.append(chunk)
            if streaming_callback is not None:
                streaming_callback(chunk)
        span.set_outputs({"response_length": len(full_response)})

    logger.info("generate_stream_node: LLM response=%s...", full_response[:200])

    return {
        "answer": full_response,
        "candidate_answers": [full_response],
        "_streaming_buffer": streaming_buffer,
        "metadata": {
            **dict(state.get("metadata", {})),
            **prompt_metadata,
            "finish_reason": getattr(llm_service, "last_finish_reason", None) or "",
            "max_tokens": max_tokens,
        },
    }


def validate_node(
    state: AgentState,
    tracer: LangSmithTracer | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    _tracer = tracer or get_tracer()
    response_text = state.get("answer", "") or ""
    sources = state.get("citations", [])

    with _tracer.trace(
        "validate_citations",
        "chain",
        inputs={"query": state.get("query", ""), "response_length": len(response_text)},
        session_id=session_id,
        user_id=user_id,
    ) as span:
        source_infos = [
            SourceInfo(
                title=str(s.get("title", "Unknown")),
                url=str(s.get("url", "")),
                domain=str(s.get("domain", "")),
                source_id=str(s.get("source_id", "")),
                citation_id=int(s.get("citation_id", 0)),
                chunk_index=int(s.get("chunk_index", 0)),
                total_chunks=int(s.get("total_chunks", 1)),
                score=float(s.get("score", 0.0)),
                snippet=str(s.get("snippet", "")),
            )
            for s in sources
        ]

        metadata = dict(state.get("metadata", {}))
        summary, body, cited_ids_in_response = validate_response(
            response=response_text,
            sources=source_infos,
            metadata=metadata,
        )
        errors = list(summary.errors)
        warnings = list(summary.warnings)
        valid_source_ids = [s.citation_id for s in source_infos]
        logger.debug("Cited IDs in response body: %s", cited_ids_in_response)
        logger.debug("Valid source citation_ids: %s", valid_source_ids)

        if not source_infos:
            confidence = 0.0
        elif "LLM indicated information not found in sources" in errors:
            confidence = 0.3
        elif (
            "No inline citations in response body" in errors
            or "Response contains citations not found in sources" in errors
            or "LLM response truncated by token cap" in errors
        ):
            confidence = 0.4
        else:
            confidence = 0.8

        if "No inline citations in response body" in errors:
            logger.warning("No inline citations in response body — appending source list")
            source_block = "\n\n**Sources:**\n"
            for s in source_infos:
                source_block += f"[{s.citation_id}] {s.title} - {s.url}\n"
            response_text += source_block
        elif "Response contains citations not found in sources" in errors:
            logger.warning("Citations missing or out of range — appending source list")
            source_block = "\n\n**Sources:**\n"
            for s in source_infos:
                source_block += f"[{s.citation_id}] {s.title} - {s.url}\n"
            response_text += source_block

        verbosity = state.get("metadata", {}).get("verbosity", VERBOSITY_CONCISE)
        response_word_count = len(body.split())
        length_warning = verbosity == VERBOSITY_CONCISE and response_word_count > CONCISE_RESPONSE_WORD_CAP
        truncated = state.get("metadata", {}).get("finish_reason") == "length"
        if length_warning:
            logger.warning(
                "Concise query answered with %d words (cap %d) — downgrading confidence",
                response_word_count,
                CONCISE_RESPONSE_WORD_CAP,
            )
            confidence = min(confidence, 0.5)
            metadata["length_warning"] = True
        if truncated:
            logger.warning("LLM response truncated by token cap (finish_reason=length)")
        metadata["truncated"] = truncated
        metadata["response_word_count"] = response_word_count
        metadata["validation_status"] = summary.status.value
        metadata["validation_warnings"] = warnings
        metadata["validation_retryable"] = summary.retryable

        citations_valid = summary.status != ValidationStatus.FAIL
        span.set_outputs(
            {
                "citations_valid": citations_valid,
                "confidence": confidence,
                "errors": errors,
                "warnings": warnings,
                "cited_ids_in_response": cited_ids_in_response,
                "response_word_count": response_word_count,
                "length_warning": length_warning,
                "truncated": truncated,
            }
        )

    return {
        "confidence": confidence,
        "citations_valid": citations_valid,
        "errors": errors,
        "answer": response_text,
        "metadata": metadata,
    }


def finalize_node(
    state: AgentState,
    tracer: LangSmithTracer | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Construct RAGResult from the final pipeline state."""
    _tracer = tracer or get_tracer()
    answer = state.get("answer", "") or ""

    with _tracer.trace(
        "finalize_result",
        "chain",
        inputs={"query": state.get("query", ""), "answer_length": len(answer)},
        session_id=session_id,
        user_id=user_id,
    ) as span:
        sources: list[SourceInfo] = []
        citation_map: dict[int, SourceInfo] = {}

        for i, c in enumerate(state.get("citations", []), start=1):
            src = SourceInfo(
                title=str(c.get("title", "Unknown")),
                url=str(c.get("url", "")),
                domain=str(c.get("domain", "")),
                source_id=str(c.get("source_id", "")),
                citation_id=int(c.get("citation_id", i)),
                chunk_index=int(c.get("chunk_index", 0)),
                total_chunks=int(c.get("total_chunks", 1)),
                score=float(c.get("score", 0.0)),
                snippet=str(c.get("snippet", "")),
            )
            sources.append(src)
            citation_map[src.citation_id] = src

        raw_chunks = [str(doc.get("page_content", doc.get("content", ""))) for doc in state.get("documents", [])]

        cited_ids = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]

        start_time = state.get("metadata", {}).get("start_time")
        latency_ms = (time.perf_counter() - start_time) * 1000 if start_time else 0.0
        cv = state.get("citations_valid")
        citations_valid = cv if isinstance(cv, bool) else True

        if not citations_valid:
            sources = []
            citation_map = {}

        result = RAGResult(
            response=answer,
            sources=sources,
            citation_map=citation_map,
            query=state.get("query", ""),
            raw_chunks=raw_chunks,
            latency_ms=latency_ms,
            total_chunks_retrieved=len(state.get("documents", [])),
            citations_valid=citations_valid,
            truncated=state.get("metadata", {}).get("finish_reason") == "length",
        )

        logger.info(
            {
                "query": result.query,
                "num_sources": len(sources),
                "latency_ms": result.latency_ms,
                "citation_ids": [s.citation_id for s in sources],
                "citations_valid": result.citations_valid,
                "cited_ids": cited_ids,
                "response_preview": answer[:300],
            }
        )

        span.set_outputs(
            {
                "response": answer,
                "citations_valid": result.citations_valid,
                "num_sources": len(sources),
                "latency_ms": result.latency_ms,
                "response_length": len(answer),
                "cited_ids": cited_ids,
                "citations": [
                    {
                        "citation_id": s.citation_id,
                        "source_id": s.source_id,
                        "title": s.title,
                        "url": s.url,
                        "domain": s.domain,
                        "score": s.score,
                    }
                    for s in sources
                ],
                "cited_ids_in_response": cited_ids,
            }
        )

    return {
        "rag_result": result,
        "metadata": {**state.get("metadata", {}), "citations_valid": result.citations_valid, "cited_ids": cited_ids},
    }

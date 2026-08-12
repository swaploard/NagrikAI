from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.state import CompiledStateGraph

from nagrik_ai.agent.agent_graph import create_agent_graph as _build_agent_graph
from nagrik_ai.agent.rag_graph import create_rag_graph as _build_rag_graph
from nagrik_ai.config.config_manager import ConfigManager
from nagrik_ai.config.config_models import (
    AUTHORITY_BONUS,
    AUTHORITY_RANKING_ENABLED,
    BM25_B,
    BM25_K1,
    CHECKPOINT_DIR,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    FETCH_K,
    HYBRID_SEARCH_ENABLED,
    LAMBDA_MULT,
    MAX_RESPONSE_TOKENS,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANKER_PROVIDER,
    RRF_K,
    TOP_K,
)
from nagrik_ai.models.agent_state import AgentState
from nagrik_ai.models.rag_result import RAGResult
from nagrik_ai.prompts.prompt_registry import CompiledPromptPipeline, load_default_prompt_pipeline
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.llm_service import BaseLLMService, create_llm_service
from nagrik_ai.services.reranker import OpenRouterReranker, Reranker
from nagrik_ai.services.tracing import LangSmithTracer
from nagrik_ai.vectorstore.bm25_retriever import BM25Retriever
from nagrik_ai.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


def create_config_manager(config_path: Path | None = None) -> ConfigManager:
    return ConfigManager(config_path)


def create_chroma_store(persist_dir: str | Path | None = None) -> ChromaStore:
    model_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        logger.info("CUDA available: embedding model will run in fp16")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)
    return ChromaStore(
        collection_name="nagrik_ai_docs",
        embeddings=embeddings,
        persist_directory=str(persist_dir or CHROMA_PERSIST_DIR),
    )


def create_llm_service_from_config(config_manager: ConfigManager | None = None) -> BaseLLMService:
    if config_manager:
        config = config_manager.load()
        return create_llm_service(
            provider=config.llm_provider,
            base_url=config.ollama_base_url if config.llm_provider == "ollama" else config.openrouter.base_url,
            model=config.ollama_model if config.llm_provider == "ollama" else config.openrouter.model,
            api_key=config.openrouter.api_key if config.llm_provider == "openrouter" else None,
            max_tokens=config.max_response_tokens,
        )
    return create_llm_service(max_tokens=MAX_RESPONSE_TOKENS)


def create_reranker() -> Reranker | None:
    if not RERANKER_ENABLED:
        logger.info("Reranker is disabled")
        return None
    if RERANKER_PROVIDER == "openrouter":
        return OpenRouterReranker(model_name=RERANKER_MODEL)
    return Reranker(model_name=RERANKER_MODEL)


def create_bm25_retriever(chroma_store: ChromaStore) -> BM25Retriever | None:
    if not HYBRID_SEARCH_ENABLED:
        return None
    return BM25Retriever(chroma_store=chroma_store, k1=BM25_K1, b=BM25_B)


def create_checkpointer(
    db_path: str | Path | None = None,
) -> SqliteSaver:
    db = Path(str(db_path or CHECKPOINT_DIR / "agent_checkpoints.db"))
    db.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3

    conn = sqlite3.connect(str(db), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    logger.info("Checkpointer initialized at %s", db)
    return saver


def create_retrieval_service(
    chroma_store: ChromaStore | None = None,
    reranker: Reranker | None = None,
) -> DocumentRetrievalService:
    store = chroma_store or create_chroma_store()
    bm25 = create_bm25_retriever(store)
    return DocumentRetrievalService(
        chroma_store=store,
        top_k=TOP_K,
        fetch_k=FETCH_K,
        lambda_mult=LAMBDA_MULT,
        reranker=reranker if reranker is not None else create_reranker(),
        hybrid_search=HYBRID_SEARCH_ENABLED and bm25 is not None,
        bm25_retriever=bm25,
        rrf_k=RRF_K,
        authority_ranking_enabled=AUTHORITY_RANKING_ENABLED,
        authority_bonus=AUTHORITY_BONUS,
    )


def create_rag_graph(
    retrieval_service: DocumentRetrievalService | None = None,
    reranker: Reranker | None = None,
    llm_service: BaseLLMService | None = None,
    tracer: LangSmithTracer | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    system_prompt: str = "",
    compiled_pipeline: CompiledPromptPipeline | None = None,
    enable_streaming: bool = False,
    enable_fallback: bool = False,
    enable_self_correction: bool = False,
    max_retries: int = 2,
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    logger.info("Creating RAG graph with wired dependencies")
    if retrieval_service is None:
        retrieval_service = create_retrieval_service()
    if llm_service is None:
        llm_service = create_llm_service(temperature=0.1, max_tokens=MAX_RESPONSE_TOKENS)
    resolved_pipeline = None if system_prompt else (compiled_pipeline or load_default_prompt_pipeline())
    resolved_reranker = reranker if reranker is not None else retrieval_service.reranker
    return _build_rag_graph(
        retrieval_service=retrieval_service,
        reranker=resolved_reranker,
        llm_service=llm_service,
        system_prompt=system_prompt,
        compiled_pipeline=resolved_pipeline,
        tracer=tracer,
        checkpointer=checkpointer,
        enable_streaming=enable_streaming,
        enable_fallback=enable_fallback,
        enable_self_correction=enable_self_correction,
        max_retries=max_retries,
    )


class _RAGGraphEntry(Protocol):
    def __call__(self, query: str, session_id: str | None = None, user_id: str | None = None) -> RAGResult: ...


class _RAGGraphStreamEntry(Protocol):
    def __call__(
        self, query: str, session_id: str | None = None, user_id: str | None = None
    ) -> Iterator[dict[str, Any]]: ...


def create_rag_graph_entry(
    retrieval_service: DocumentRetrievalService | None = None,
    reranker: Reranker | None = None,
    llm_service: BaseLLMService | None = None,
    tracer: LangSmithTracer | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    system_prompt: str = "",
    compiled_pipeline: CompiledPromptPipeline | None = None,
    enable_fallback: bool = False,
    enable_self_correction: bool = False,
    max_retries: int = 2,
) -> tuple[CompiledStateGraph[AgentState, Any, Any, Any], _RAGGraphEntry, _RAGGraphStreamEntry]:
    """Returns (graph, run_query, stream_query) with all dependencies wired."""
    from nagrik_ai.agent.rag_graph import run_rag_query, stream_rag_query

    def _run_query(query: str, session_id: str | None = None, user_id: str | None = None) -> RAGResult:
        return run_rag_query(
            query=query,
            session_id=session_id,
            user_id=user_id,
            retrieval_service=retrieval_service,
            reranker=reranker,
            llm_service=llm_service,
            system_prompt=system_prompt,
            compiled_pipeline=compiled_pipeline,
            tracer=tracer,
            checkpointer=checkpointer,
            enable_fallback=enable_fallback,
            enable_self_correction=enable_self_correction,
            max_retries=max_retries,
        )

    def _stream_query(
        query: str, session_id: str | None = None, user_id: str | None = None
    ) -> Iterator[dict[str, Any]]:
        return stream_rag_query(
            query=query,
            session_id=session_id,
            user_id=user_id,
            retrieval_service=retrieval_service,
            reranker=reranker,
            llm_service=llm_service,
            system_prompt=system_prompt,
            compiled_pipeline=compiled_pipeline,
            tracer=tracer,
            checkpointer=checkpointer,
            enable_fallback=enable_fallback,
            enable_self_correction=enable_self_correction,
            max_retries=max_retries,
        )

    graph = create_rag_graph(
        retrieval_service=retrieval_service or create_retrieval_service(),
        reranker=reranker,
        llm_service=llm_service,
        system_prompt=system_prompt,
        compiled_pipeline=compiled_pipeline,
        tracer=tracer,
        checkpointer=checkpointer,
        enable_fallback=enable_fallback,
        enable_self_correction=enable_self_correction,
        max_retries=max_retries,
    )
    return graph, _run_query, _stream_query


class AppStack(NamedTuple):
    """Single instance of every dependency, built once at startup."""

    graph: CompiledStateGraph[AgentState, Any, Any, Any]
    run_query: _RAGGraphEntry
    stream_query: _RAGGraphStreamEntry
    retrieval_service: DocumentRetrievalService
    reranker: Reranker | None
    llm_service: BaseLLMService
    chroma_store: ChromaStore


_stack: AppStack | None = None


def build_app_stack(
    retrieval_service: DocumentRetrievalService | None = None,
    reranker: Reranker | None = None,
    llm_service: BaseLLMService | None = None,
    tracer: LangSmithTracer | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    system_prompt: str = "",
    compiled_pipeline: CompiledPromptPipeline | None = None,
    persist_dir: str | Path | None = None,
    enable_fallback: bool = False,
    enable_self_correction: bool = False,
    max_retries: int = 2,
) -> AppStack:
    """Build the full app stack once; every dependency is a single shared instance.

    The returned graph is compiled with streaming enabled, so both `run_query` and
    `stream_query` can execute against the same prebuilt graph.
    """
    from nagrik_ai.agent.rag_graph import run_query_from_graph, stream_query_from_graph

    global _stack
    if _stack is not None:
        return _stack

    chroma_store = retrieval_service.chroma_store if retrieval_service is not None else create_chroma_store(persist_dir)
    resolved_reranker = reranker
    if resolved_reranker is None:
        resolved_reranker = retrieval_service.reranker if retrieval_service is not None else create_reranker()
    resolved_retrieval = retrieval_service or create_retrieval_service(
        chroma_store=chroma_store,
        reranker=resolved_reranker,
    )
    resolved_llm = llm_service or create_llm_service(temperature=0.1, max_tokens=MAX_RESPONSE_TOKENS)

    graph = create_rag_graph(
        retrieval_service=resolved_retrieval,
        reranker=resolved_reranker,
        llm_service=resolved_llm,
        system_prompt=system_prompt,
        compiled_pipeline=compiled_pipeline,
        tracer=tracer,
        checkpointer=checkpointer,
        enable_streaming=True,
        enable_fallback=enable_fallback,
        enable_self_correction=enable_self_correction,
        max_retries=max_retries,
    )

    def _run_query(query: str, session_id: str | None = None, user_id: str | None = None) -> RAGResult:
        return run_query_from_graph(
            graph,
            query,
            session_id=session_id,
            user_id=user_id,
        )

    def _stream_query(
        query: str, session_id: str | None = None, user_id: str | None = None
    ) -> Iterator[dict[str, Any]]:
        return stream_query_from_graph(
            graph,
            query,
            session_id=session_id,
            user_id=user_id,
        )

    stack = AppStack(
        graph=graph,
        run_query=_run_query,
        stream_query=_stream_query,
        retrieval_service=resolved_retrieval,
        reranker=resolved_reranker,
        llm_service=resolved_llm,
        chroma_store=chroma_store,
    )
    _stack = stack
    logger.info("Built app stack with single shared graph and dependencies")
    return stack


def get_app_stack(
    persist_dir: str | Path | None = None,
    enable_fallback: bool = False,
    enable_self_correction: bool = False,
    max_retries: int = 2,
) -> AppStack:
    """Return the prebuilt app stack, building it lazily on first call if needed."""
    if _stack is None:
        return build_app_stack(
            persist_dir=persist_dir,
            enable_fallback=enable_fallback,
            enable_self_correction=enable_self_correction,
            max_retries=max_retries,
        )
    return _stack


def warmup(stack: AppStack) -> AppStack:
    """Absorb cold-start costs once at boot: embed query, BM25 index build, rerank predict."""
    start = time.perf_counter()

    try:
        stack.chroma_store.embeddings.embed_query("nagrik-ai warmup query")
        logger.info("Warmup: embedding model embed_query complete")
    except Exception:
        logger.exception("Warmup: embedding model embed_query failed")

    bm25 = stack.retrieval_service.bm25_retriever
    if bm25 is not None:
        try:
            bm25._build_index()
            logger.info("Warmup: BM25 index built with %d documents", len(bm25._documents))
        except Exception:
            logger.exception("Warmup: BM25 index build failed")
    else:
        logger.info("Warmup: BM25 index skipped (hybrid search disabled)")

    reranker = stack.reranker
    if reranker is not None and not isinstance(reranker, OpenRouterReranker):
        try:
            reranker.rerank(
                "nagrik-ai warmup query",
                [{"content": "warmup document for tuning inference", "metadata": {}}],
            )
            logger.info("Warmup: reranker first predict complete")
        except Exception:
            logger.exception("Warmup: reranker predict failed")
    else:
        logger.info("Warmup: reranker predict skipped (disabled or OpenRouter)")

    total_elapsed = (time.perf_counter() - start) * 1000
    logger.info("Warmup complete in %.0f ms", total_elapsed)
    return stack


def create_agent_graph(
    llm_service: BaseLLMService | None = None,
    tracer: LangSmithTracer | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    logger.info("Creating agent graph with wired dependencies")
    _tracer = tracer  # Reserved for future tracing integration
    if llm_service is None:
        llm_service = create_llm_service()
    return _build_agent_graph(
        llm_service=llm_service,
        checkpointer=checkpointer,
    )

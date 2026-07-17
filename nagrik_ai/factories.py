from __future__ import annotations

import logging
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

from nagrik_ai.config.config_manager import ConfigManager
from nagrik_ai.config.config_models import (
    BM25_B,
    BM25_K1,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    FETCH_K,
    HYBRID_SEARCH_ENABLED,
    LAMBDA_MULT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RRF_K,
    TOP_K,
)
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.llm_service import LLMService
from nagrik_ai.services.rag_orchestrator import RAGOrchestrator
from nagrik_ai.services.reranker import Reranker
from nagrik_ai.vectorstore.bm25_retriever import BM25Retriever
from nagrik_ai.vectorstore.chroma_store import ChromaStore


def create_config_manager(config_path: Path | None = None) -> ConfigManager:
    return ConfigManager(config_path)


def create_chroma_store(persist_dir: str | Path | None = None) -> ChromaStore:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return ChromaStore(
        collection_name="nagrik_ai_docs",
        embeddings=embeddings,
        persist_directory=str(persist_dir or CHROMA_PERSIST_DIR),
    )


def create_llm_service() -> LLMService:
    return LLMService(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)


def create_reranker() -> Reranker | None:
    if not RERANKER_ENABLED:
        logger.info("Reranker is disabled")
        return None
    return Reranker(model_name=RERANKER_MODEL)


def create_bm25_retriever(chroma_store: ChromaStore) -> BM25Retriever | None:
    if not HYBRID_SEARCH_ENABLED:
        return None
    return BM25Retriever(chroma_store=chroma_store, k1=BM25_K1, b=BM25_B)


def create_retrieval_service(chroma_store: ChromaStore | None = None) -> DocumentRetrievalService:
    store = chroma_store or create_chroma_store()
    bm25 = create_bm25_retriever(store)
    return DocumentRetrievalService(
        chroma_store=store,
        top_k=TOP_K,
        fetch_k=FETCH_K,
        lambda_mult=LAMBDA_MULT,
        reranker=create_reranker(),
        hybrid_search=HYBRID_SEARCH_ENABLED and bm25 is not None,
        bm25_retriever=bm25,
        rrf_k=RRF_K,
    )


def create_orchestrator(
    retrieval_service: DocumentRetrievalService | None = None,
    llm_service: LLMService | None = None,
) -> RAGOrchestrator:
    logger.info("Initializing RAG orchestrator")
    return RAGOrchestrator(
        retrieval_service=retrieval_service or create_retrieval_service(),
        llm_service=llm_service or create_llm_service(),
    )


class RAGOrchestratorFactory:
    """Factory for creating RAGOrchestrator instances with all dependencies wired together."""

    def __init__(
        self,
        collection_name: str = "nagrik_ai_docs",
        persist_directory: str | None = None,
        embedding_model: str = EMBEDDING_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
        ollama_model: str = OLLAMA_MODEL,
        top_k: int = TOP_K,
        fetch_k: int = FETCH_K,
        lambda_mult: float = LAMBDA_MULT,
        reranker_model: str = RERANKER_MODEL,
        reranker_enabled: bool = RERANKER_ENABLED,
        hybrid_search_enabled: bool = HYBRID_SEARCH_ENABLED,
        bm25_k1: float = BM25_K1,
        bm25_b: float = BM25_B,
        rrf_k: int = RRF_K,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(CHROMA_PERSIST_DIR)
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.reranker_model = reranker_model
        self.reranker_enabled = reranker_enabled
        self.hybrid_search_enabled = hybrid_search_enabled
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.rrf_k = rrf_k

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    def create_chroma_store(self, embeddings: HuggingFaceEmbeddings | None = None) -> ChromaStore:
        if embeddings is None:
            embeddings = self.create_embeddings()
        return ChromaStore(
            collection_name=self.collection_name,
            embeddings=embeddings,
            persist_directory=self.persist_directory,
        )

    def create_reranker(self) -> Reranker | None:
        if not self.reranker_enabled:
            logger.info("Reranker is disabled")
            return None
        return Reranker(model_name=self.reranker_model)

    def create_bm25_retriever(self, chroma_store: ChromaStore) -> BM25Retriever | None:
        if not self.hybrid_search_enabled:
            return None
        return BM25Retriever(
            chroma_store=chroma_store,
            k1=self.bm25_k1,
            b=self.bm25_b,
        )

    def create_document_retrieval_service(self, chroma_store: ChromaStore | None = None) -> DocumentRetrievalService:
        if chroma_store is None:
            chroma_store = self.create_chroma_store()
        bm25 = self.create_bm25_retriever(chroma_store)
        return DocumentRetrievalService(
            chroma_store=chroma_store,
            top_k=self.top_k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
            reranker=self.create_reranker(),
            hybrid_search=self.hybrid_search_enabled and bm25 is not None,
            bm25_retriever=bm25,
            rrf_k=self.rrf_k,
        )

    def create_llm_service(self) -> LLMService:
        return LLMService(base_url=self.ollama_base_url, model=self.ollama_model)

    def create_orchestrator(self) -> RAGOrchestrator:
        embeddings = self.create_embeddings()
        chroma_store = self.create_chroma_store(embeddings)
        doc_service = self.create_document_retrieval_service(chroma_store)
        llm_service = self.create_llm_service()
        return RAGOrchestrator(
            retrieval_service=doc_service,
            llm_service=llm_service,
        )

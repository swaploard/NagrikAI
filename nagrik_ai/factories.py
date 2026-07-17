from __future__ import annotations

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from nagrik_ai.config.config_manager import ConfigManager
from nagrik_ai.config.settings import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOP_K,
)
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.llm_service import LLMService
from nagrik_ai.services.rag_orchestrator import RAGOrchestrator
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


def create_retrieval_service(chroma_store: ChromaStore | None = None) -> DocumentRetrievalService:
    return DocumentRetrievalService(
        chroma_store=chroma_store or create_chroma_store(),
        top_k=TOP_K,
    )


def create_orchestrator(
    retrieval_service: DocumentRetrievalService | None = None,
    llm_service: LLMService | None = None,
) -> RAGOrchestrator:
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
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(CHROMA_PERSIST_DIR)
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.top_k = top_k

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

    def create_document_retrieval_service(self, chroma_store: ChromaStore | None = None) -> DocumentRetrievalService:
        if chroma_store is None:
            chroma_store = self.create_chroma_store()
        return DocumentRetrievalService(chroma_store=chroma_store, top_k=self.top_k)

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

from __future__ import annotations

from pathlib import Path

from nagrik_ai.config.config_manager import ConfigManager
from nagrik_ai.config.settings import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOP_K,
)
from nagrik_ai.prompts.prompt_loader import PromptLoader
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.llm_service import LLMService
from nagrik_ai.services.rag_orchestrator import RAGOrchestrator
from nagrik_ai.vectorstore.chroma_store import ChromaStore


def create_config_manager(config_path: Path | None = None) -> ConfigManager:
    return ConfigManager(config_path)


def create_chroma_store(persist_dir: str | Path | None = None) -> ChromaStore:
    return ChromaStore(
        persist_dir=persist_dir or CHROMA_PERSIST_DIR,
        model_name=EMBEDDING_MODEL,
    )


def create_llm_service() -> LLMService:
    return LLMService(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)


def create_retrieval_service(chroma_store: ChromaStore | None = None) -> DocumentRetrievalService:
    return DocumentRetrievalService(
        chroma_store=chroma_store or create_chroma_store(),
        top_k=TOP_K,
    )


def create_prompt_loader(prompt_dir: Path | None = None) -> PromptLoader:
    return PromptLoader(prompt_dir)


def create_orchestrator(
    retrieval_service: DocumentRetrievalService | None = None,
    llm_service: LLMService | None = None,
    prompt_loader: PromptLoader | None = None,
) -> RAGOrchestrator:
    return RAGOrchestrator(
        retrieval_service=retrieval_service or create_retrieval_service(),
        llm_service=llm_service or create_llm_service(),
        prompt_loader=prompt_loader or create_prompt_loader(),
    )

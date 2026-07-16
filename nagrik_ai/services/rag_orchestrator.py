from __future__ import annotations

from collections.abc import Iterator

from nagrik_ai.prompts.prompt_loader import load_prompt
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.llm_service import LLMService


class RAGOrchestrator:
    def __init__(
        self,
        retrieval_service: DocumentRetrievalService,
        llm_service: LLMService,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

    def query(self, user_query: str) -> str:
        results = self.retrieval_service.retrieve(user_query)
        context = self.retrieval_service.format_context(results)
        system_prompt = load_prompt("system_prompt")
        user_prompt = load_prompt("user_query", question=user_query, context=context)
        return self.llm_service.generate(user_prompt, system=system_prompt)

    def query_stream(self, user_query: str) -> Iterator[str]:
        results = self.retrieval_service.retrieve(user_query)
        context = self.retrieval_service.format_context(results)
        system_prompt = load_prompt("system_prompt")
        user_prompt = load_prompt("user_query", question=user_query, context=context)
        return self.llm_service.generate_stream(user_prompt, system=system_prompt)

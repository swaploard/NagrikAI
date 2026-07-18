from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from nagrik_ai.prompts.prompt_loader import load_prompt
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.llm_service import BaseLLMService

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG query containing answer and source citations."""
    answer: str
    citations: dict[int, dict[str, Any]]  # citation_num -> {source_id, title, url, domain, content, ...}


class RAGOrchestrator:
    def __init__(
        self,
        retrieval_service: DocumentRetrievalService,
        llm_service: BaseLLMService,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service
        logger.info("Initialized RAG orchestrator")

    def query(self, user_query: str) -> RAGResponse:
        results = self.retrieval_service.retrieve(user_query)
        context, citation_mapping = self.retrieval_service.format_context(results)
        system_prompt = load_prompt("system_prompt")
        user_prompt = load_prompt("user_query", question=user_query, context=context)
        answer = self.llm_service.generate(user_prompt, system=system_prompt)
        return RAGResponse(answer=answer, citations=citation_mapping)

    def query_stream(self, user_query: str) -> Iterator[RAGResponse]:
        results = self.retrieval_service.retrieve(user_query)
        context, citation_mapping = self.retrieval_service.format_context(results)
        system_prompt = load_prompt("system_prompt")
        user_prompt = load_prompt("user_query", question=user_query, context=context)

        first_chunk = True
        for chunk in self.llm_service.generate_stream(user_prompt, system=system_prompt):
            if first_chunk:
                yield RAGResponse(answer=chunk, citations=citation_mapping)
                first_chunk = False
            else:
                yield RAGResponse(answer=chunk, citations={})


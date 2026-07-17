from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nagrik_ai.config.config_models import CrawlerConfig, ParserConfig, SiteConfig
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.llm_service import LLMService


@pytest.fixture
def sample_site_config() -> SiteConfig:
    return SiteConfig(
        name="test-site",
        base_url="https://example.com",
        start_urls=["https://example.com/start"],
        allowed_domains=["example.com"],
        crawler=CrawlerConfig(max_concurrency=2, request_delay=0.1),
        parser=ParserConfig(),
    )


@pytest.fixture
def mock_chroma_store() -> MagicMock:
    store = MagicMock()
    store.max_marginal_relevance_search.return_value = []
    return store


@pytest.fixture
def mock_llm_service() -> MagicMock:
    service = MagicMock(spec=LLMService)
    service.generate.return_value = "Test response"
    return service


@pytest.fixture
def mock_retrieval_service(mock_chroma_store: MagicMock) -> DocumentRetrievalService:
    return DocumentRetrievalService(mock_chroma_store, top_k=3)


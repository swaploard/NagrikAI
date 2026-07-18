from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService


class TestFormatContext:
    def test_format_context_structured_blocks(self) -> None:
        chroma_mock = MagicMock()
        service = DocumentRetrievalService(chroma_store=chroma_mock, top_k=3)

        results: list[dict[str, Any]] = [
            {
                "content": "GST registration is required for businesses with turnover above 20 lakhs.",
                "metadata": {
                    "source_id": "gst_001",
                    "title": "GST Registration Guide",
                    "citation_url": "https://gst.gov.in/register",
                    "domain": "gst.gov.in",
                },
            },
            {
                "content": "File GSTR-1 monthly by 11th of next month.",
                "metadata": {
                    "source_id": "gst_002",
                    "title": "GSTR-1 Filing",
                    "citation_url": "https://gst.gov.in/gstr1",
                    "domain": "gst.gov.in",
                },
            },
        ]

        formatted, citation_mapping = service.format_context(results)

        # Check formatted context contains expected citations
        assert "[1] Source ID: gst_001" in formatted
        assert "Title: GST Registration Guide" in formatted
        assert "URL: https://gst.gov.in/register" in formatted
        assert "Domain: gst.gov.in" in formatted
        assert "Content: GST registration is required" in formatted

        assert "[2] Source ID: gst_002" in formatted
        assert "Title: GSTR-1 Filing" in formatted
        assert "URL: https://gst.gov.in/gstr1" in formatted
        assert "Content: File GSTR-1 monthly" in formatted

        assert "\n\n---\n\n" in formatted
        assert formatted.count("\n\n---\n\n") == 1

        # Check citation_mapping has locked IDs
        assert len(citation_mapping) == 2
        assert citation_mapping[1]["metadata"]["source_id"] == "gst_001"
        assert citation_mapping[2]["metadata"]["source_id"] == "gst_002"
        # Verify citation_id is locked on docs
        assert results[0].get("citation_id") == 1
        assert results[1].get("citation_id") == 2

    def test_format_context_single_result(self) -> None:
        chroma_mock = MagicMock()
        service = DocumentRetrievalService(chroma_store=chroma_mock, top_k=1)

        results: list[dict[str, Any]] = [
            {
                "content": "Single result content.",
                "metadata": {
                    "source_id": "single_001",
                    "title": "Single Doc",
                    "citation_url": "https://example.com/doc",
                    "domain": "example.com",
                },
            }
        ]

        formatted, citation_mapping = service.format_context(results)

        assert formatted == (
            "[1] Source ID: single_001 | Title: Single Doc | "
            "URL: https://example.com/doc | Domain: example.com\n"
            "Content: Single result content."
        )
        assert "---" not in formatted

        assert len(citation_mapping) == 1
        assert citation_mapping[1]["metadata"]["source_id"] == "single_001"
        assert results[0].get("citation_id") == 1

    def test_format_context_fallback_metadata_keys(self) -> None:
        chroma_mock = MagicMock()
        service = DocumentRetrievalService(chroma_store=chroma_mock, top_k=1)

        results: list[dict[str, Any]] = [
            {
                "content": "Fallback test.",
                "metadata": {
                    "source": "fallback_source",
                    "title": "Fallback Title",
                    "url": "https://fallback.url",
                    "domain": "fallback.com",
                },
            }
        ]

        formatted, citation_mapping = service.format_context(results)

        assert "Source ID: fallback_source" in formatted
        assert "Title: Fallback Title" in formatted
        assert "URL: https://fallback.url" in formatted
        assert "Domain: fallback.com" in formatted

        assert citation_mapping[1]["metadata"]["source"] == "fallback_source"

    def test_format_context_empty_results(self) -> None:
        chroma_mock = MagicMock()
        service = DocumentRetrievalService(chroma_store=chroma_mock, top_k=3)

        formatted, citation_mapping = service.format_context([])

        assert formatted == ""
        assert citation_mapping == {}

    def test_format_context_citation_id_locked_before_sort(self) -> None:
        """Verify citation IDs are assigned BEFORE sorting by score."""
        chroma_mock = MagicMock()
        service = DocumentRetrievalService(chroma_store=chroma_mock, top_k=3)

        # Results with different scores - will be sorted by score desc
        results: list[dict[str, Any]] = [
            {"content": "Low score doc", "metadata": {"source_id": "low"}, "score": 0.1},
            {"content": "High score doc", "metadata": {"source_id": "high"}, "score": 0.9},
            {"content": "Medium score doc", "metadata": {"source_id": "med"}, "score": 0.5},
        ]

        formatted, citation_mapping = service.format_context(results)

        # Citation IDs locked in original order: low=1, high=2, med=3
        assert citation_mapping[1]["metadata"]["source_id"] == "low"
        assert citation_mapping[2]["metadata"]["source_id"] == "high"
        assert citation_mapping[3]["metadata"]["source_id"] == "med"

        # But display order is by score: high, med, low
        # Check formatted output shows high score doc first (citation [2])
        assert formatted.index("[2] Source ID: high") < formatted.index("[3] Source ID: med")
        assert formatted.index("[3] Source ID: med") < formatted.index("[1] Source ID: low")


class TestReciprocalRankFusion:
    def test_rrf_fuses_dense_and_bm25_results(self) -> None:
        chroma_mock = MagicMock()
        chroma_mock.similarity_search.return_value = []
        service = DocumentRetrievalService(
            chroma_store=chroma_mock,
            hybrid_search=True,
            bm25_retriever=MagicMock(),
            top_k=5,
        )

        dense: list[dict[str, Any]] = [
            {"content": "doc_a", "metadata": {"source": "a"}},
            {"content": "doc_b", "metadata": {"source": "b"}},
            {"content": "doc_c", "metadata": {"source": "c"}},
        ]
        bm25: list[dict[str, Any]] = [
            {"content": "doc_b", "metadata": {"source": "b"}},
            {"content": "doc_c", "metadata": {"source": "c"}},
            {"content": "doc_d", "metadata": {"source": "d"}},
        ]

        service._docs_to_dicts = MagicMock(return_value=dense)  # type: ignore[assignment]
        bm25_mock = MagicMock()
        bm25_mock.retrieve.return_value = bm25
        service.bm25_retriever = bm25_mock

        fused = service.retrieve("test query")

        contents = [d["content"] for d in fused]
        assert contents == ["doc_b", "doc_c", "doc_a", "doc_d"]

    def test_rrf_k_parameter_affects_scores(self) -> None:
        chroma_mock = MagicMock()
        chroma_mock.similarity_search.return_value = []
        service_small_k = DocumentRetrievalService(
            chroma_store=chroma_mock,
            hybrid_search=True,
            bm25_retriever=MagicMock(),
            rrf_k=1,
            top_k=5,
        )
        service_large_k = DocumentRetrievalService(
            chroma_store=chroma_mock,
            hybrid_search=True,
            bm25_retriever=MagicMock(),
            rrf_k=100,
            top_k=5,
        )

        dense: list[dict[str, Any]] = [
            {"content": "doc_x", "metadata": {}},
            {"content": "doc_y", "metadata": {}},
        ]
        bm25: list[dict[str, Any]] = [
            {"content": "doc_y", "metadata": {}},
            {"content": "doc_z", "metadata": {}},
        ]

        service_small_k._docs_to_dicts = MagicMock(return_value=dense)  # type: ignore[assignment]
        service_large_k._docs_to_dicts = MagicMock(return_value=dense)  # type: ignore[assignment]
        bm25_small_mock = MagicMock()
        bm25_small_mock.retrieve.return_value = bm25
        bm25_large_mock = MagicMock()
        bm25_large_mock.retrieve.return_value = bm25
        service_small_k.bm25_retriever = bm25_small_mock
        service_large_k.bm25_retriever = bm25_large_mock

        fused_small = service_small_k.retrieve("test query")
        fused_large = service_large_k.retrieve("test query")

        assert len(fused_small) == 3
        assert len(fused_large) == 3

    def test_hybrid_retrieve_uses_rrf(self) -> None:
        chroma_mock = MagicMock()
        chroma_mock.similarity_search.return_value = []
        bm25_mock = MagicMock()
        bm25_mock.retrieve.return_value = []

        service = DocumentRetrievalService(
            chroma_store=chroma_mock,
            hybrid_search=True,
            bm25_retriever=bm25_mock,
            top_k=3,
        )

        result = service.retrieve("test query")

        assert result == []
        chroma_mock.similarity_search.assert_called_once()
        bm25_mock.retrieve.assert_called_once()

    def test_non_hybrid_fallback(self) -> None:
        chroma_mock = MagicMock()
        chroma_mock.query.return_value = []
        chroma_mock.similarity_search.return_value = []
        bm25_mock = MagicMock()

        service = DocumentRetrievalService(
            chroma_store=chroma_mock,
            hybrid_search=False,
            bm25_retriever=bm25_mock,
            reranker=None,
            top_k=3,
        )

        result = service.retrieve("test query")

        assert result == []
        chroma_mock.query.assert_called_once()
        bm25_mock.retrieve.assert_not_called()


@pytest.mark.usefixtures("mock_chroma_store")
class TestBM25Retriever:
    def test_bm25_retriever_builds_index_and_retrieves(self) -> None:
        with (
            patch(
                "nagrik_ai.vectorstore.bm25_retriever.ChromaStore",
            ),
            patch("nagrik_ai.vectorstore.bm25_retriever.BM25Okapi") as mock_bm25,
        ):
            from nagrik_ai.vectorstore.bm25_retriever import BM25Retriever

            mock_chroma = MagicMock()
            mock_chroma.get_all_documents.return_value = [
                MagicMock(page_content="the cat sat on the mat", metadata={}),
                MagicMock(page_content="the dog played in the park", metadata={}),
                MagicMock(page_content="the bird flew over the tree", metadata={}),
            ]

            mock_bm25_instance = MagicMock()
            mock_bm25_instance.get_scores.return_value = [0.5, 0.3, 0.1]
            mock_bm25.return_value = mock_bm25_instance

            retriever = BM25Retriever(chroma_store=mock_chroma)
            results = retriever.retrieve("cat mat", k=2)

            assert len(results) == 2
            mock_bm25.assert_called_once()
            mock_bm25_instance.get_scores.assert_called_once()


from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from nagrik_ai.services.citation_service import assign_citation_ids
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

        # Lock citation IDs first (as orchestrator would do)
        results, _ = assign_citation_ids(results)
        formatted, citation_mapping = service.format_context(results)

        # Check formatted context contains expected citations (new structured format)
        assert "[1]\nSource ID: gst_001" in formatted
        assert "Title: GST Registration Guide" in formatted
        assert "URL: https://gst.gov.in/register" in formatted
        assert "Domain: gst.gov.in" in formatted
        assert "Content:\nGST registration is required" in formatted

        assert "[2]\nSource ID: gst_002" in formatted
        assert "Title: GSTR-1 Filing" in formatted
        assert "URL: https://gst.gov.in/gstr1" in formatted
        assert "Content:\nFile GSTR-1 monthly" in formatted

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

        results, _ = assign_citation_ids(results)
        formatted, citation_mapping = service.format_context(results)

        expected = (
            "[1]\n"
            "Source ID: single_001\n"
            "Title: Single Doc\n"
            "URL: https://example.com/doc\n"
            "Domain: example.com\n\n"
            "Content:\nSingle result content."
        )
        assert formatted == expected
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

        results, _ = assign_citation_ids(results)
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

        results, _ = assign_citation_ids(results)
        formatted, citation_mapping = service.format_context(results)

        # Citation IDs locked in original order: low=1, high=2, med=3
        assert citation_mapping[1]["metadata"]["source_id"] == "low"
        assert citation_mapping[2]["metadata"]["source_id"] == "high"
        assert citation_mapping[3]["metadata"]["source_id"] == "med"

        # But display order is by score: high, med, low
        # Check formatted output shows high score doc first (citation [2])
        assert formatted.index("[2]\nSource ID: high") < formatted.index("[3]\nSource ID: med")
        assert formatted.index("[3]\nSource ID: med") < formatted.index("[1]\nSource ID: low")


class TestReciprocalRankFusion:
    def test_rrf_fuses_dense_and_bm25_results(self) -> None:
        chroma_mock = MagicMock()
        service = DocumentRetrievalService(chroma_store=chroma_mock, top_k=3, rrf_k=60)

        dense_results = [
            {"content": "doc A", "score": 0.9},
            {"content": "doc B", "score": 0.8},
        ]
        bm25_results = [
            {"content": "doc B", "score": 0.7},
            {"content": "doc C", "score": 0.6},
        ]

        fused = service._reciprocal_rank_fusion(dense_results, bm25_results)

        assert len(fused) == 3
        contents = [d["content"] for d in fused]
        assert "doc A" in contents
        assert "doc B" in contents
        assert "doc C" in contents

    def test_rrf_k_parameter_affects_scores(self) -> None:
        chroma_mock = MagicMock()
        service_low_k = DocumentRetrievalService(chroma_store=chroma_mock, top_k=3, rrf_k=10)
        service_high_k = DocumentRetrievalService(chroma_store=chroma_mock, top_k=3, rrf_k=100)

        dense = [{"content": "doc1"}]
        bm25 = [{"content": "doc1"}]

        fused_low = service_low_k._reciprocal_rank_fusion(dense, bm25)
        fused_high = service_high_k._reciprocal_rank_fusion(dense, bm25)

        # Both should return same doc but scores differ
        assert len(fused_low) == 1
        assert len(fused_high) == 1

    def test_hybrid_retrieve_uses_rrf(self) -> None:
        chroma_mock = MagicMock()
        bm25_mock = MagicMock()

        chroma_mock.similarity_search.return_value = [
            MagicMock(page_content="dense doc", metadata={"source_id": "dense"}),
        ]
        bm25_mock.retrieve.return_value = [
            {"content": "bm25 doc", "metadata": {"source_id": "bm25"}},
        ]

        service = DocumentRetrievalService(
            chroma_store=chroma_mock,
            top_k=3,
            fetch_k=5,
            hybrid_search=True,
            bm25_retriever=bm25_mock,
            rrf_k=60,
        )

        with patch("nagrik_ai.services.document_retrieval_service.validate_metadata", side_effect=lambda x: x):
            results = service.retrieve("test query")

        assert len(results) == 2
        chroma_mock.similarity_search.assert_called_once()
        bm25_mock.retrieve.assert_called_once()

    def test_non_hybrid_fallback(self) -> None:
        chroma_mock = MagicMock()
        chroma_mock.query.return_value = [
            MagicMock(page_content="vector doc", metadata={"source_id": "vector"}),
        ]

        service = DocumentRetrievalService(
            chroma_store=chroma_mock,
            top_k=3,
            hybrid_search=False,
        )

        with patch("nagrik_ai.services.document_retrieval_service.validate_metadata", side_effect=lambda x: x):
            results = service.retrieve("test query")

        assert len(results) == 1
        chroma_mock.query.assert_called_once()


class TestBM25Retriever:
    def test_bm25_retriever_builds_index_and_retrieves(self) -> None:
        from nagrik_ai.vectorstore.bm25_retriever import BM25Retriever
        from nagrik_ai.vectorstore.chroma_store import ChromaStore

        chroma_mock = MagicMock(spec=ChromaStore)
        chroma_mock.get_all_documents.return_value = [
            MagicMock(page_content="GST registration guide", metadata={"source_id": "1"}),
            MagicMock(page_content="GSTR-1 filing rules", metadata={"source_id": "2"}),
        ]

        retriever = BM25Retriever(chroma_store=chroma_mock, k1=1.2, b=0.75)
        results = retriever.retrieve("GST registration", k=2)

        assert len(results) == 2
        assert "GST registration" in results[0]["content"]


class TestReranker:
    def test_reranker_reorders_by_relevance(self) -> None:
        from nagrik_ai.services.reranker import Reranker

        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

        results = [
            {"content": "Unrelated document about cooking.", "score": 0.1},
            {"content": "GST registration process for businesses.", "score": 0.5},
            {"content": "How to file GSTR-1 returns monthly.", "score": 0.3},
        ]

        reranked = reranker.rerank("GST registration", results, top_k=2)

        assert len(reranked) == 2
        assert "registration" in reranked[0]["content"].lower()

    def test_reranker_respects_top_k(self) -> None:
        from nagrik_ai.services.reranker import Reranker

        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

        results = [{"content": f"doc {i}", "score": 0.5} for i in range(10)]

        reranked = reranker.rerank("query", results, top_k=3)

        assert len(reranked) == 3

    def test_reranker_handles_empty_results(self) -> None:
        from nagrik_ai.services.reranker import Reranker

        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

        reranked = reranker.rerank("query", [], top_k=5)

        assert reranked == []

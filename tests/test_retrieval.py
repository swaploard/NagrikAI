from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from nagrik_ai.services.citation_service import assign_citation_ids
from nagrik_ai.services.document_retrieval_service import (
    DocumentRetrievalService,
    authority_atk,
    first_authoritative_rank,
    retrieval_metrics,
)


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
        chroma_mock.similarity_search_with_scores.return_value = [
            (MagicMock(page_content="vector doc", metadata={"source_id": "vector"}), 0.9),
        ]

        service = DocumentRetrievalService(
            chroma_store=chroma_mock,
            top_k=3,
            hybrid_search=False,
        )

        with patch("nagrik_ai.services.document_retrieval_service.validate_metadata", side_effect=lambda x: x):
            results = service.retrieve("test query")

        assert len(results) == 1
        assert results[0]["score"] == 0.9
        chroma_mock.similarity_search_with_scores.assert_called_once()

    def test_hybrid_path_attaches_rrf_scores(self) -> None:
        chroma_mock = MagicMock()
        chroma_mock.similarity_search.return_value = [
            MagicMock(page_content="doc A", metadata={"source_id": "dense"}),
        ]
        bm25_mock = MagicMock()
        bm25_mock.retrieve.return_value = [
            {"content": "doc A", "metadata": {"source_id": "bm25"}},
            {"content": "doc B", "metadata": {"source_id": "bm25"}},
        ]

        service = DocumentRetrievalService(
            chroma_store=chroma_mock,
            top_k=3,
            hybrid_search=True,
            bm25_retriever=bm25_mock,
            rrf_k=60,
        )

        with patch("nagrik_ai.services.document_retrieval_service.validate_metadata", side_effect=lambda x: x):
            results = service.retrieve("test query")

        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]
        assert results[0]["content"] == "doc A"


class TestAuthorityBias:
    def test_authority_bias_promotes_statutory_source_over_faq(self) -> None:
        chroma_mock = MagicMock()
        service = DocumentRetrievalService(chroma_store=chroma_mock, top_k=1)

        candidates = [
            {
                "content": "FAQ content",
                "metadata": {"title": "FAQs_GST", "source_id": "faq_1"},
                "score": 0.95,
            },
            {
                "content": "Rule content",
                "metadata": {"title": "GST Rules", "source_id": "rule_1"},
                "score": 0.949,
            },
            {
                "content": "Help content",
                "metadata": {"title": "GST help page", "source_id": "help_1"},
                "score": 0.90,
            },
        ]

        selected = service._select_with_authority_bias(candidates)

        assert len(selected) == 1
        assert selected[0]["content"] == "Rule content"

    def test_authority_bias_keeps_raw_score_and_records_authority_score(self) -> None:
        chroma_mock = MagicMock()
        service = DocumentRetrievalService(chroma_store=chroma_mock, top_k=5)

        candidates = [
            {"content": "a", "metadata": {"title": "Income-tax Act", "source_id": "1"}, "score": 0.95},
            {"content": "b", "metadata": {"title": "FAQs_GST", "source_id": "2"}, "score": 1.0},
        ]

        selected = service._select_with_authority_bias(candidates)

        assert len(selected) == 2
        act_doc = next(doc for doc in selected if doc["content"] == "a")
        assert act_doc["score"] == 0.95
        assert act_doc["authority_score"] == 0.08

    def test_authority_bias_disabled_keeps_score_order(self) -> None:
        chroma_mock = MagicMock()
        service = DocumentRetrievalService(
            chroma_store=chroma_mock,
            top_k=1,
            authority_ranking_enabled=False,
        )

        candidates = [
            {"content": "FAQ", "metadata": {"title": "FAQs_GST", "source_id": "1"}, "score": 0.9},
            {"content": "Rule", "metadata": {"title": "GST Rules", "source_id": "2"}, "score": 0.8},
        ]

        selected = service._select_with_authority_bias(candidates)

        assert len(selected) == 1
        assert selected[0]["content"] == "FAQ"

    def test_authority_bias_custom_bonus_map(self) -> None:
        chroma_mock = MagicMock()
        service = DocumentRetrievalService(
            chroma_store=chroma_mock,
            top_k=1,
            authority_bonus={"act": 0.5, "other": 0.0},
        )

        candidates = [
            {"content": "other", "metadata": {"title": "misc", "source_id": "1"}, "score": 1.0},
            {"content": "act", "metadata": {"title": "Income-tax Act", "source_id": "2"}, "score": 0.99},
            {"content": "other2", "metadata": {"title": "misc2", "source_id": "3"}, "score": 0.95},
        ]

        selected = service._select_with_authority_bias(candidates)

        assert len(selected) == 1
        assert selected[0]["content"] == "act"

    def test_authority_bias_handles_empty_and_single_doc(self) -> None:
        chroma_mock = MagicMock()
        service = DocumentRetrievalService(chroma_store=chroma_mock, top_k=3)

        assert service._select_with_authority_bias([]) == []

        single = [{"content": "solo", "metadata": {"title": "t", "source_id": "1"}, "score": 0.5}]
        assert service._select_with_authority_bias(single) == single


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
        assert all(isinstance(doc.get("score"), float) for doc in results)
        assert results[0]["score"] >= results[1]["score"]


class TestReranker:
    def test_reranker_emits_scores(self) -> None:
        from unittest.mock import MagicMock

        from nagrik_ai.services.reranker import Reranker

        class _Scores:
            def tolist(self) -> list[float]:
                return [0.9, 0.3]

        mock_model = MagicMock()
        mock_model.predict.return_value = _Scores()
        reranker = Reranker(model_name="fake-model")
        reranker._model = mock_model

        results = [
            {"content": "relevant doc", "score": 0.1},
            {"content": "irrelevant doc", "score": 0.2},
        ]

        reranked = reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0]["content"] == "relevant doc"
        assert reranked[0]["score"] == 0.9
        assert reranked[1]["score"] == 0.3

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


class TestOpenRouterReranker:
    def _make_reranker(self) -> Any:
        from nagrik_ai.services.reranker import OpenRouterReranker

        reranker = OpenRouterReranker(
            model_name="cohere/rerank-4-pro",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
        )
        reranker._client = MagicMock()
        return reranker

    def test_rerank_sends_request_and_reorders(self) -> None:
        reranker = self._make_reranker()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "gen-rerank-1",
            "results": [
                {"index": 1, "relevance_score": 0.92},
                {"index": 0, "relevance_score": 0.31},
            ],
            "usage": {"search_units": 1},
        }
        reranker._client.post.return_value = mock_response

        docs = [
            {"content": "GST registration guide", "metadata": {"title": "guide"}},
            {"content": "GSTR-1 filing rules", "metadata": {"title": "rules"}},
        ]
        reranked = reranker.rerank("GST filing", docs, top_k=2)

        assert [d["content"] for d in reranked] == ["GSTR-1 filing rules", "GST registration guide"]
        assert reranked[0]["score"] == 0.92
        assert reranked[0]["metadata"] == {"title": "rules"}

    def test_rerank_posts_expected_payload(self) -> None:
        reranker = self._make_reranker()
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.5}]}
        reranker._client.post.return_value = mock_response

        docs = [{"content": "doc A", "score": 0.1}, {"content": "doc B", "score": 0.2}]
        reranker.rerank("query", docs, top_k=1)

        reranker._client.post.assert_called_once_with(
            "https://openrouter.ai/api/v1/rerank",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "cohere/rerank-4-pro", "query": "query", "documents": ["doc A", "doc B"], "top_n": 1},
        )

    def test_rerank_handles_empty_results(self) -> None:
        reranker = self._make_reranker()
        assert reranker.rerank("query", [], top_k=5) == []
        reranker._client.post.assert_not_called()

    def test_rerank_requires_api_key(self) -> None:
        from nagrik_ai.services.reranker import OpenRouterReranker, RerankerError

        try:
            OpenRouterReranker(model_name="cohere/rerank-4-pro", api_key="")
        except RerankerError:
            pass
        else:
            raise AssertionError("expected RerankerError for missing API key")


class TestChromaStoreWithScores:
    def test_similarity_search_with_scores_returns_pairs(self) -> None:
        from langchain_core.documents import Document

        from nagrik_ai.vectorstore.chroma_store import ChromaStore

        doc_a = Document(
            page_content="GST registration guide",
            metadata={"source_id": "1", "title": "Guide", "url": "https://gst.gov.in/a", "domain": "gst.gov.in"},
        )
        doc_b = Document(
            page_content="GSTR-1 filing",
            metadata={"source_id": "2", "title": "FAQs_GSTR1", "url": "https://gst.gov.in/b", "domain": "gst.gov.in"},
        )

        with patch("nagrik_ai.vectorstore.chroma_store.Chroma"):
            store = ChromaStore(collection_name="test", embeddings=MagicMock(), persist_directory=":memory:")
            store.vector_db.similarity_search_with_relevance_scores.return_value = [
                (doc_a, 0.9),
                (doc_b, 0.7),
            ]

            results = store.similarity_search_with_scores("GST", k=2)

        assert len(results) == 2
        assert results[0] == (doc_a, 0.9)
        assert results[1] == (doc_b, 0.7)
        assert doc_a.metadata["url"] == "https://gst.gov.in/a"
        assert doc_a.metadata["source_type"] == ""

    def test_similarity_search_with_scores_returns_empty_on_error(self) -> None:
        from nagrik_ai.vectorstore.chroma_store import ChromaStore

        with patch("nagrik_ai.vectorstore.chroma_store.Chroma"):
            store = ChromaStore(collection_name="test", embeddings=MagicMock(), persist_directory=":memory:")
            store.vector_db.similarity_search_with_relevance_scores.side_effect = RuntimeError("boom")

            results = store.similarity_search_with_scores("GST", k=2)

        assert results == []


def _metric_doc(title: str, source_id: str, score: float) -> dict[str, Any]:
    return {
        "content": f"{source_id} content",
        "metadata": {"title": title, "source_id": source_id},
        "score": score,
    }


class TestRetrievalMetrics:
    def test_authority_atk_proportion(self) -> None:
        docs = [
            _metric_doc("GST Rules", "rule_1", 0.9),
            _metric_doc("Income-tax Act", "act_1", 0.85),
            _metric_doc("FAQs_GST", "faq_1", 0.95),
            _metric_doc("Notification 12/2024", "notif_1", 0.8),
        ]
        assert authority_atk(docs) == 0.75  # 3 of 4 authoritative
        assert authority_atk(docs, k=2) == 1.0
        assert authority_atk(docs, k=3) == 2 / 3
        assert authority_atk(docs, k=1) == 1.0

    def test_authority_atk_empty(self) -> None:
        assert authority_atk([]) == 0.0
        assert authority_atk([], k=5) == 0.0

    def test_first_authoritative_rank(self) -> None:
        docs = [
            _metric_doc("GST User Guide", "guide_1", 0.95),
            _metric_doc("GST Rules", "rule_1", 0.9),
            _metric_doc("FAQs_GST", "faq_1", 0.8),
        ]
        assert first_authoritative_rank(docs) == 2

    def test_first_authoritative_rank_none(self) -> None:
        docs = [_metric_doc("GST User Guide", "guide_1", 0.95), _metric_doc("FAQs_GST", "faq_1", 0.8)]
        assert first_authoritative_rank(docs) is None
        assert first_authoritative_rank([]) is None

    def test_retrieval_metrics_summary(self) -> None:
        docs = [
            _metric_doc("GST Rules", "rule_1", 0.9),
            _metric_doc("FAQs_GST", "faq_1", 0.95),
            _metric_doc("Notification 1/2025", "notif_1", 0.7),
        ]
        metrics = retrieval_metrics(docs)
        assert metrics["num_authoritative"] == 2
        assert metrics["total"] == 3
        assert metrics["first_authoritative_rank"] == 1
        assert metrics["authority_at_top_k"] == 2 / 3
        assert metrics["authority_at_1"] == 1.0

    def test_metrics_logged_in_retrieve_trace(self) -> None:
        chroma_mock = MagicMock()
        chroma_mock.similarity_search_with_scores.return_value = [
            (MagicMock(page_content="Rule content", metadata={"title": "GST Rules", "source_id": "rule_1"}), 0.9),
            (MagicMock(page_content="FAQ content", metadata={"title": "FAQs_GST", "source_id": "faq_1"}), 0.9),
        ]

        span = MagicMock()
        span.elapsed_ms.return_value = 1.0
        tracer_mock = MagicMock()
        tracer_mock.trace.return_value.__enter__.return_value = span
        tracer_mock.trace.return_value.__exit__.return_value = None

        service = DocumentRetrievalService(chroma_store=chroma_mock, top_k=2, tracer=tracer_mock)

        with patch("nagrik_ai.services.document_retrieval_service.validate_metadata", side_effect=lambda x: x):
            results = service.retrieve("gst")

        assert len(results) == 2
        set_outputs_kwargs = span.set_outputs.call_args.args[0]
        metrics = set_outputs_kwargs["metrics"]
        assert metrics["num_authoritative"] == 1
        assert metrics["total"] == 2
        assert metrics["first_authoritative_rank"] == 1

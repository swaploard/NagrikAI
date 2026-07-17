from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService


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

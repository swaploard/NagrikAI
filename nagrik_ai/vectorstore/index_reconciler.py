from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from langchain_core.documents import Document

from nagrik_ai.vectorstore.bm25_retriever import BM25Retriever
from nagrik_ai.vectorstore.chroma_store import ChromaStore
from nagrik_ai.vectorstore.index_outbox import IndexOperation, IndexOutbox

logger = logging.getLogger(__name__)


class IndexBackend(Protocol):
    def apply(self, operation: IndexOperation) -> None: ...

    def inspect(self, operation: IndexOperation) -> dict[str, set[str]]: ...


ChunkProvider = Callable[[IndexOperation], list[Document]]


class RetrievalIndexBackend:
    """Idempotently applies desired state to Chroma and invalidates BM25."""

    def __init__(
        self,
        chroma_store: ChromaStore,
        chunk_provider: ChunkProvider,
        bm25_retriever: BM25Retriever | None = None,
    ) -> None:
        self.chroma_store = chroma_store
        self.chunk_provider = chunk_provider
        self.bm25_retriever = bm25_retriever

    def apply(self, operation: IndexOperation) -> None:
        chunks = self.chunk_provider(operation)
        by_id = {str(chunk.metadata.get("chunk_id", "")): chunk for chunk in chunks}
        if set(by_id) != set(operation.expected_chunk_ids) or "" in by_id:
            raise ValueError("chunk provider did not return the expected stable chunk IDs")
        desired_scopes = sorted(set(operation.desired_scopes))
        for chunk in chunks:
            chunk.metadata["retrieval_scopes"] = desired_scopes
            chunk.metadata["document_id"] = operation.document_id
            chunk.metadata["version_id"] = operation.target_version_id
        existing = self.inspect(operation)
        stale_ids = sorted(set(existing) - set(operation.expected_chunk_ids))
        self.chroma_store.delete_chunks(stale_ids)
        ordered_ids = sorted(by_id)
        self.chroma_store.add_documents([by_id[chunk_id] for chunk_id in ordered_ids], ids=ordered_ids)
        if self.bm25_retriever is not None:
            self.bm25_retriever.invalidate()

    def inspect(self, operation: IndexOperation) -> dict[str, set[str]]:
        return self.chroma_store.index_state(operation.document_id, operation.target_version_id)


class IndexReconciler:
    def __init__(self, outbox: IndexOutbox, backend: IndexBackend) -> None:
        self.outbox = outbox
        self.backend = backend

    def process_pending(self) -> int:
        completed = 0
        for operation in self.outbox.pending():
            try:
                self.backend.apply(operation)
                observed = self.backend.inspect(operation)
                self.outbox.mark_complete(operation, observed)
                completed += 1
            except Exception as exc:
                logger.exception("Index operation %s remains replayable", operation.operation_id)
                self.outbox.mark_retry(operation.operation_id, str(exc))
        return completed

    def reconcile(self) -> int:
        repaired = 0
        for operation in self.outbox.desired_operations():
            try:
                observed = self.backend.inspect(operation)
                expected_scopes = set(operation.desired_scopes)
                consistent = set(observed) == set(operation.expected_chunk_ids) and all(
                    scopes == expected_scopes for scopes in observed.values()
                )
                if not consistent:
                    self.backend.apply(operation)
                    observed = self.backend.inspect(operation)
                    repaired += 1
                self.outbox.mark_complete(operation, observed)
            except Exception as exc:
                logger.exception("Could not reconcile index operation %s", operation.operation_id)
                self.outbox.mark_retry(operation.operation_id, str(exc))
        return repaired

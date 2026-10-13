"""Vectorize markdown content into ChromaDB using LangChain components."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document

from nagrik_ai.utils.markdown_utils import find_markdown_files, read_markdown_file
from nagrik_ai.vectorstore.chroma_store import ChromaStore, validate_metadata
from nagrik_ai.vectorstore.index_outbox import IndexFingerprint, IndexOperation, IndexOutbox
from nagrik_ai.vectorstore.index_reconciler import IndexReconciler, RetrievalIndexBackend
from nagrik_ai.vectorstore.legal_chunker import LegalChunker


class TextSplitter(Protocol):
    def split_documents(self, documents: list[Document]) -> list[Document]: ...


logger = logging.getLogger(__name__)


class MarkdownVectorizer:
    """Vectorize markdown content and store in ChromaDB using LangChain.

    This class handles the vectorization pipeline for markdown documents,
    including text splitting and storage in a vector database. Dependencies
    are injected to enable testing and configuration flexibility.
    """

    def __init__(
        self,
        chroma_store: ChromaStore,
        text_splitter: TextSplitter,
        legal_chunker: LegalChunker | None = None,
        embedding_provider: str = "unknown",
        embedding_model: str = "unknown",
        embedding_model_version: str = "unknown",
        index_outbox: IndexOutbox | None = None,
    ) -> None:
        self.chroma_store = chroma_store
        self.text_splitter = text_splitter
        self.legal_chunker = legal_chunker or LegalChunker()
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_model_version = embedding_model_version
        self.index_outbox = index_outbox
        if index_outbox is not None:
            self.chroma_store.index_db_path = index_outbox.path

        logger.debug("Initialized MarkdownVectorizer")

    def process_directory(
        self,
        input_dir: str,
        site_filter: str | None = None,
        batch_size: int = 20,
    ) -> int:
        markdown_files = [
            path
            for path in find_markdown_files(input_dir, site_filter)
            if Path(path).name != "index.md"
        ]
        total_files = len(markdown_files)

        logger.info("Found %d markdown files to process", total_files)

        processed_count = 0
        chunk_count = 0
        for i in range(0, total_files, batch_size):
            batch = markdown_files[i : i + batch_size]
            new_chunks = self._process_batch(batch)
            processed_count += len(batch)
            chunk_count += new_chunks
            logger.info(
                "Processed %d/%d files (%d chunks)",
                processed_count,
                total_files,
                chunk_count,
            )

        return processed_count

    def _process_batch(self, file_paths: list[str]) -> int:
        all_documents: list[Document] = []

        for file_path in file_paths:
            try:
                metadata, content = read_markdown_file(file_path)

                if not content:
                    logger.warning("Empty content in %s", file_path)
                    continue

                doc = Document(
                    page_content=content,
                    metadata=self._prepare_metadata(metadata, file_path),
                )

                chunks = self._split(doc, content)

                self._finalize_chunks(chunks, file_path)

                all_documents.extend(chunks)

                logger.debug(
                    "Added document %s with embeddings",
                    Path(file_path).name,
                )

            except Exception:
                logger.exception("Error processing file %s", file_path)

        if all_documents:
            self._store_chunks(all_documents)

        return len(all_documents)

    def _normalize_metadata(self, metadata: Any) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            normalized[key] = value
        return normalized

    def _prepare_metadata(
        self,
        metadata: dict[str, Any],
        file_path: str,
    ) -> dict[str, Any]:
        doc_id = Path(file_path).stem

        normalized_metadata = self._normalize_metadata(metadata)
        enriched_metadata = normalized_metadata.copy()

        enriched_metadata.setdefault("source_id", doc_id)
        enriched_metadata.setdefault("content_hash", hashlib.sha256(Path(file_path).read_bytes()).hexdigest())
        enriched_metadata["source_path"] = file_path
        enriched_metadata["file_name"] = Path(file_path).name

        if "source_url" in normalized_metadata:
            enriched_metadata["source_url"] = normalized_metadata["source_url"]
            enriched_metadata["url"] = normalized_metadata["source_url"]
        elif "url" in normalized_metadata:
            enriched_metadata["source_url"] = normalized_metadata["url"]

        if "source_url" in enriched_metadata:
            enriched_metadata["citation_url"] = enriched_metadata["source_url"]
        elif "url" in enriched_metadata:
            enriched_metadata["citation_url"] = enriched_metadata["url"]

        return enriched_metadata

    def process_file(self, file_path: str) -> int:
        try:
            metadata, content = read_markdown_file(file_path)

            if not content:
                logger.warning("Empty content in %s", file_path)
                return 0

            doc = Document(
                page_content=content,
                metadata=self._prepare_metadata(metadata, file_path),
            )

            chunks = self._split(doc, content)

            self._finalize_chunks(chunks, file_path)

            self._store_chunks(chunks)

            logger.debug(
                "Added document %s with embeddings",
                Path(file_path).name,
            )

            return len(chunks)

        except Exception:
            logger.exception("Error processing file %s", file_path)
            return 0

    def _split(self, document: Document, content: str) -> list[Document]:
        metadata = self._normalize_metadata(document.metadata)
        if metadata.get("document_id") and metadata.get("version_id"):
            return self.legal_chunker.chunk_markdown(content, metadata)
        return self.text_splitter.split_documents([document])

    def _store_chunks(self, chunks: list[Document]) -> None:
        outbox = self.index_outbox
        if outbox is None:
            self.chroma_store.add_documents(chunks, ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks])
            return
        groups: dict[tuple[str, str], list[Document]] = {}
        for chunk in chunks:
            metadata = chunk.metadata
            document_id = str(metadata.get("document_id") or metadata["source_id"])
            version_id = str(metadata.get("version_id") or metadata.get("content_hash") or "legacy")
            groups.setdefault((document_id, version_id), []).append(chunk)
        for (document_id, version_id), group in groups.items():
            metadata = group[0].metadata
            raw_scopes = metadata.get("retrieval_scopes", '["all_versions", "current_source"]')
            scopes = json.loads(raw_scopes) if isinstance(raw_scopes, str) else raw_scopes
            fingerprint = IndexFingerprint(
                content_hash=str(metadata.get("content_hash") or ""),
                schema_version=str(metadata.get("schema_version") or "legacy"),
                parser_name=str(metadata.get("parser_name") or "Parser"),
                parser_version=str(metadata.get("parser_version") or "legacy"),
                extraction_parser_name=str(metadata.get("extraction_parser_name") or ""),
                extraction_parser_version=str(metadata.get("extraction_parser_version") or ""),
                chunker_name=self.legal_chunker.name,
                chunker_version=self.legal_chunker.version,
                embedding_provider=self.embedding_provider,
                embedding_model=self.embedding_model,
                embedding_model_version=self.embedding_model_version,
            )
            outbox.plan(
                document_id=document_id,
                target_version_id=version_id,
                desired_scopes=scopes,
                expected_chunk_ids=[str(chunk.metadata["chunk_id"]) for chunk in group],
                fingerprint=fingerprint,
                chunk_payloads=[{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in group],
            )
        reconciler = self.index_reconciler()
        reconciler.process_pending()
        if outbox.pending():
            raise RuntimeError("Index operations remain pending; rerun vectorize to replay them")

    def index_reconciler(self) -> IndexReconciler:
        outbox = self.index_outbox
        if outbox is None:
            raise RuntimeError("Transactional indexing is not configured")

        def provide(operation: IndexOperation) -> list[Document]:
            return [Document.model_validate(payload) for payload in outbox.chunk_payloads(operation.operation_id)]

        return IndexReconciler(outbox, RetrievalIndexBackend(self.chroma_store, provide))

    def _add_lineage(self, metadata: dict[str, Any]) -> None:
        metadata["embedding_provider"] = self.embedding_provider
        metadata["embedding_model"] = self.embedding_model
        metadata["embedding_model_version"] = self.embedding_model_version
        metadata.setdefault("chunker_name", self.legal_chunker.name)
        metadata.setdefault("chunker_version", self.legal_chunker.version)

    def _finalize_chunks(self, chunks: list[Document], file_path: str) -> None:
        """Attach complete, stable identity and lineage to every chunk."""
        for index, chunk in enumerate(chunks):
            metadata = self._normalize_metadata(getattr(chunk, "metadata", None))
            metadata["chunk_index"] = index
            metadata["total_chunks"] = len(chunks)
            location = str(metadata.get("logical_location") or f"chunk:{index}")
            metadata["logical_location"] = location
            if not metadata.get("chunk_id"):
                document_id = str(
                    metadata.get("document_id")
                    or metadata.get("source_id")
                    or metadata.get("source_path")
                    or Path(file_path).resolve()
                )
                version_id = str(metadata.get("version_id") or metadata.get("content_hash") or "")
                metadata["chunk_id"] = self.legal_chunker.chunk_id(
                    document_id,
                    version_id,
                    location,
                )
            self._add_lineage(metadata)
            chunk.metadata = validate_metadata(metadata)

"""Vectorize markdown content into ChromaDB using LangChain components."""

import logging
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document

from nagrik_ai.utils.markdown_utils import find_markdown_files, read_markdown_file
from nagrik_ai.vectorstore.chroma_store import ChromaStore, validate_metadata


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
    ) -> None:
        self.chroma_store = chroma_store
        self.text_splitter = text_splitter

        logger.debug("Initialized MarkdownVectorizer")

    def process_directory(
        self,
        input_dir: str,
        site_filter: str | None = None,
        batch_size: int = 20,
    ) -> int:
        markdown_files = find_markdown_files(input_dir, site_filter)
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

                chunks = self.text_splitter.split_documents([doc])

                for i, chunk in enumerate(chunks):
                    md = self._normalize_metadata(getattr(chunk, "metadata", None))
                    md["chunk_index"] = i
                    md["total_chunks"] = len(chunks)
                    chunk.metadata = validate_metadata(md)

                all_documents.extend(chunks)

                logger.debug(
                    "Added document %s with embeddings",
                    Path(file_path).name,
                )

            except Exception:
                logger.exception("Error processing file %s", file_path)

        if all_documents:
            self.chroma_store.add_documents(all_documents)

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

        enriched_metadata["source_id"] = doc_id
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

        return validate_metadata(enriched_metadata)

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

            chunks = self.text_splitter.split_documents([doc])

            for i, chunk in enumerate(chunks):
                md = self._normalize_metadata(getattr(chunk, "metadata", None))
                md["chunk_index"] = i
                md["total_chunks"] = len(chunks)
                chunk.metadata = validate_metadata(md)

            self.chroma_store.add_documents(chunks)

            logger.debug(
                "Added document %s with embeddings",
                Path(file_path).name,
            )

            return len(chunks)

        except Exception:
            logger.exception("Error processing file %s", file_path)
            return 0

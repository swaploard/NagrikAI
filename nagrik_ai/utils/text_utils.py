"""Utilities for text and HTML processing."""

import logging
import re
from typing import Any, cast

from langchain_core.documents import Document
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from nagrik_ai.config.config_models import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

MIN_CHUNKABLE_CONTENT_LENGTH = 100


def is_pdf_url(url: str) -> bool:
    return url.lower().endswith(".pdf") or "pdf" in url.rsplit("/", maxsplit=1)[-1].lower()


def chunk_html_content(
    html_content: str,
    content_type: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    splitter_type: str = "semantic",
    max_chunks: int = 50,
) -> list[dict[str, Any]]:
    if chunk_overlap >= chunk_size:
        msg = f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        raise ValueError(msg)

    html_content = remove_javascript(html_content)

    if not html_content or len(html_content.strip()) < MIN_CHUNKABLE_CONTENT_LENGTH:
        logger.warning("Content is too short or empty, not chunking")
        return [{"content": html_content.strip(), "metadata": {}}]

    plain_text = re.sub(r"<[^>]+>", " ", html_content)
    plain_text = re.sub(r"\s+", " ", plain_text).strip()

    if len(plain_text) <= chunk_size:
        logger.info(
            "Content size (%d chars) is smaller than chunk_size (%d), skipping chunking",
            len(plain_text),
            chunk_size,
        )
        return [{"content": plain_text, "metadata": {}}]

    estimated_chunks = len(plain_text) // (chunk_size - chunk_overlap) + 1
    if estimated_chunks > max_chunks:
        logger.warning(
            "Content would generate too many chunks (%d). Using simpler chunking method.",
            estimated_chunks,
        )
        return _chunk_text_safely(plain_text, chunk_size, chunk_overlap, max_chunks)

    if not content_type or "text/html" not in content_type.lower():
        return _chunk_text_safely(plain_text, chunk_size, chunk_overlap, max_chunks)

    try:
        if splitter_type == "header":
            logger.info("Using header-based HTML chunking")
            headers_to_split_on = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
                ("h5", "Header 5"),
            ]
            html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
            split_docs = html_splitter.split_text(html_content)

        elif splitter_type == "section":
            logger.info("Using section-based HTML chunking")
            headers_to_split_on = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
                ("h5", "Header 5"),
            ]
            section_splitter: HTMLHeaderTextSplitter = HTMLSectionSplitter(headers_to_split_on)  # type: ignore[assignment]
            split_docs = section_splitter.split_text(html_content)

        else:
            logger.info("Using recursive character-based text chunking for HTML")
            cleaned_html = _basic_clean_html(html_content)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            split_docs = text_splitter.create_documents([cleaned_html])

        if len(split_docs) > max_chunks:
            logger.warning(
                "HTML splitting produced too many chunks (%d). Limiting to %d.",
                len(split_docs),
                max_chunks,
            )
            split_docs = split_docs[:max_chunks]

        return [
            {
                "content": doc.page_content,
                "metadata": _normalize_metadata(getattr(doc, "metadata", None)),
            }
            for doc in split_docs
        ]

    except Exception:
        logger.warning("Error using HTML splitter. Falling back to default chunker.", exc_info=True)
        return _chunk_text_safely(plain_text, chunk_size, chunk_overlap, max_chunks)


def remove_javascript(html_content: str) -> str:
    cleaned = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL)
    cleaned = re.sub(r' on\w+="[^"]*"', "", cleaned)
    cleaned = re.sub(r" on\w+='[^']*'", "", cleaned)
    cleaned = re.sub(r'href="javascript:[^"]*"', 'href="#"', cleaned)
    cleaned = re.sub(r"href='javascript:[^']*'", "href='#'", cleaned)
    return re.sub(r"(\s)javascript:", r"\1", cleaned)


def _normalize_metadata(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, dict):
        normalized: dict[str, Any] = {}
        for key, value in cast(dict[Any, Any], metadata).items():
            if isinstance(key, str):
                normalized[key] = value
        return normalized
    return {}


def _chunk_text_safely(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_chunks: int = 50,
) -> list[dict[str, Any]]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    try:
        docs = text_splitter.create_documents([text])

        if len(docs) > max_chunks:
            logger.warning(
                "Generated too many chunks (%d). Limiting to %d.",
                len(docs),
                max_chunks,
            )
            docs = docs[:max_chunks]

        return [{"content": doc.page_content, "metadata": {}} for doc in docs]
    except Exception:
        logger.exception("Error chunking text")
        chunks: list[dict[str, Any]] = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i : i + chunk_size]
            chunks.append({"content": chunk, "metadata": {}})
            if len(chunks) >= max_chunks:
                break
        return chunks


class HybridMarkdownSplitter:
    """Two-pass hybrid markdown splitter.

    Pass 1 (Structural): Uses MarkdownHeaderTextSplitter to split the document
    by logical sections, attaching the header hierarchy to the metadata.

    Pass 2 (Size-controlled): Runs those resulting sections through
    RecursiveCharacterTextSplitter with a defined chunk_size and chunk_overlap.

    This ensures final chunks are appropriately sized for vector embeddings
    while retaining their complete structural parent-context in the metadata.
    """

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: list[str] | None = None,
        add_start_index: bool = True,
    ) -> None:
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
            ("####", "H4"),
        ]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )
        self._size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
            or [
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n```\n",
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                " ",
                "",
            ],
            add_start_index=True,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []

        for doc in documents:
            sections = self._header_splitter.split_text(doc.page_content)

            for idx, section in enumerate(sections):
                metadata: dict[str, Any] = {
                    **_normalize_metadata(getattr(doc, "metadata", None)),
                    **_normalize_metadata(getattr(section, "metadata", None)),
                    "section_id": idx,
                }
                section.metadata = metadata

            sub_chunks = self._size_splitter.split_documents(sections)

            for chunk in sub_chunks:
                metadata = _normalize_metadata(getattr(chunk, "metadata", None))
                metadata.pop("chunk_index", None)
                metadata.pop("total_chunks", None)
                chunk.metadata = metadata

            all_chunks.extend(sub_chunks)

        return all_chunks


def _basic_clean_html(html_content: str) -> str:
    cleaned = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL)
    cleaned = re.sub(r"<style[^>]*>.*?</style>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n\n# \1\n\n", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n\n## \1\n\n", cleaned, flags=re.DOTALL)
    cleaned = re.sub(
        r"<h3[^>]*>(.*?)</h3>",
        r"\n\n### \1\n\n",
        cleaned,
        flags=re.DOTALL,
    )
    cleaned = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\n\1\n\n", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<br[^>]*>", "\n", cleaned)
    cleaned = re.sub(r"<li[^>]*>(.*?)</li>", r"\n• \1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\n\s+\n", "\n\n", cleaned)

    return cleaned.strip()

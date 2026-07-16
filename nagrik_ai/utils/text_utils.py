from __future__ import annotations

import re
from collections.abc import Iterator

from langchain.text_splitter import RecursiveCharacterTextSplitter


def clean_html_content(html: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", html)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def chunk_markdown(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


def chunk_markdown_iter(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Iterator[str]:
    yield from chunk_markdown(text, chunk_size, chunk_overlap)

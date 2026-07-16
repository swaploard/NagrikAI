"""Utilities for working with markdown content."""

import logging
from pathlib import Path

import frontmatter

logger = logging.getLogger(__name__)


def read_markdown_file(file_path: str) -> tuple[dict[str, object], str]:
    try:
        with Path(file_path).open(encoding="utf-8") as f:
            post = frontmatter.load(f)
            metadata = post.metadata
            content = post.content

            if "source_url" in metadata:
                metadata["url"] = metadata["source_url"]
            elif "source_file" in metadata:
                source_file = metadata["source_file"]
                if isinstance(source_file, str):
                    metadata["url"] = source_file

            return metadata, content
    except Exception:
        logger.exception("Error reading markdown file %s", file_path)
        return {}, ""


def find_markdown_files(directory: str, site_filter: str | None = None) -> list[str]:
    markdown_files: list[str] = []

    try:
        for file_path_obj in Path(directory).rglob("*.md"):
            file_path = str(file_path_obj)

            if site_filter:
                try:
                    relative_path = file_path_obj.parent.relative_to(directory)
                    path_parts = relative_path.parts

                    if len(path_parts) > 0 and path_parts[0] == site_filter:
                        markdown_files.append(file_path)
                except (ValueError, IndexError):
                    pass
            else:
                markdown_files.append(file_path)
    except Exception:
        logger.exception("Error finding markdown files")

    return markdown_files

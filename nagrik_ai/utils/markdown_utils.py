from __future__ import annotations

from pathlib import Path


def save_markdown(content: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def list_markdown_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.md"))

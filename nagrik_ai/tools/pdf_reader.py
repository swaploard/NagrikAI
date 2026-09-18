"""Extract PDF evidence and preserve its document-level authority metadata."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from nagrik_ai.models.tool_result import SourceAuthority, ToolResult
from nagrik_ai.tools.result_utils import failure, source_id, tool_result

MAX_OUTPUT_CHARS = 50000
MAX_FILE_BYTES = 25 * 1024 * 1024


@tool_result
def read_pdf(file_path: str) -> ToolResult:
    path = Path(file_path)
    if not path.exists():
        return failure("PDF file not found.", "NOT_FOUND")
    if not path.is_file() or path.suffix.lower() != ".pdf":
        return failure("Path must refer to a PDF file.")
    if path.stat().st_size > MAX_FILE_BYTES:
        return failure("PDF exceeds the 25 MB size limit.")
    reader = PdfReader(path)
    metadata = reader.metadata or {}
    # Explicit document metadata only; titles and file names cannot establish authority.
    declared = metadata.get("/SourceAuthority", metadata.get("/source_authority", "secondary"))
    if declared not in {"authoritative", "secondary", "general_web"}:
        return failure("Invalid PDF document authority metadata.")
    authority: SourceAuthority = declared
    parts: list[str] = []
    count = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
        count += len(text) + 2
        if count > MAX_OUTPUT_CHARS:
            break
    text = "\n\n".join(parts)[:MAX_OUTPUT_CHARS]
    if not text.strip():
        return failure("PDF contains no extractable text.", "NOT_FOUND")
    sid = source_id("pdf", path.read_bytes())
    document = {
        "source_id": sid,
        "source_authority": authority,
        "source_type": metadata.get("/SourceType", "pdf"),
        "text": text,
    }
    return ToolResult(
        True,
        {
            "document": document,
            "sources": [{"source_id": sid, "citation_id": 1, "title": str(metadata.get("/Title") or path.name)}],
        },
        None,
        None,
        authority,
        None,
        (sid,),
    )

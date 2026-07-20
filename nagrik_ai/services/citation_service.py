"""Citation service for managing source citations in RAG pipeline."""

import html
import re
from collections import OrderedDict
from typing import Any

from nagrik_ai.models.rag_result import SourceInfo


def normalize_url(url: str) -> str:
    """Normalize URL for deduplication (strip fragment & query)."""
    return url.split("#")[0].split("?")[0]


def flatten_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested metadata into top-level keys for citation formatting."""
    metadata = doc.get("metadata", {})
    return {
        "content": doc.get("content", ""),
        "page_content": doc.get("content", ""),
        "source_id": metadata.get("source_id", metadata.get("source", "unknown")),
        "title": metadata.get("title", "Unknown"),
        "url": metadata.get("citation_url", metadata.get("url", "unknown")),
        "domain": metadata.get("domain", "unknown"),
        "chunk_index": metadata.get("chunk_index", 0),
        "total_chunks": metadata.get("total_chunks", 1),
        "score": doc.get("score", 0.0),
        "citation_id": doc.get("citation_id", 1),
    }


def _make_source_info(doc: dict[str, Any], citation_id: int) -> SourceInfo:
    """Create SourceInfo from a flattened document."""
    return SourceInfo(
        title=str(doc.get("title", "Unknown")),
        url=str(doc.get("url", "")),
        domain=str(doc.get("domain", "")),
        source_id=str(doc.get("source_id", "")),
        citation_id=citation_id,
        chunk_index=int(doc.get("chunk_index", 0)),
        total_chunks=int(doc.get("total_chunks", 1)),
        score=float(doc.get("score", 0.0)),
        snippet="",
    )


def assign_citation_ids(docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, SourceInfo]]:
    """Lock citation IDs before generation — never recompute after dedup/filtering."""
    mapping: dict[int, SourceInfo] = {}
    for i, doc in enumerate(docs, start=1):
        doc["citation_id"] = i
        flat = flatten_doc(doc)
        mapping[i] = _make_source_info(flat, i)
    return docs, mapping


def deduplicate_for_display(sources: list[SourceInfo]) -> list[SourceInfo]:
    """Deduplicate by (normalized_url, title, chunk_index); keep first-occurrence citation_id."""
    seen: dict[tuple[str, str, int], SourceInfo] = OrderedDict()
    for s in sources:
        key = (normalize_url(s.url), s.title, s.chunk_index)
        if key not in seen:
            seen[key] = s
    return list(seen.values())


def format_context_block(doc: dict[str, object], index: int) -> str:
    """Format a single document as a structured context block with hard separator."""
    return (
        f"[{index}]\n"
        f"Source ID: {doc.get('source_id', '')}\n"
        f"Title: {doc.get('title', '')}\n"
        f"URL: {doc.get('url', '')}\n"
        f"Domain: {doc.get('domain', '')}\n\n"
        f"Content:\n{doc.get('page_content', doc.get('content', ''))}"
    )


def validate_citations(response: str, sources: list[SourceInfo]) -> bool:
    """Check all cited IDs exist in sources and at least one citation exists."""
    cited = set(map(int, re.findall(r"\[(\d+)\]", response)))
    valid_ids = {s.citation_id for s in sources}
    return cited.issubset(valid_ids) and len(cited) > 0


def extract_snippet(text: str, query: str, max_len: int = 160) -> str:
    """Extract the sentence most relevant to the query (sentence-boundary aware)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    best = None
    best_score = -1
    for sent in sentences:
        score = 0
        for term in query.lower().split():
            if term in sent.lower():
                score += 1
        if score > best_score:
            best_score = score
            best = sent
    if best and len(best) <= max_len:
        return best
    return (best or text)[:max_len]


def make_citations_clickable(response: str, sources: list[SourceInfo]) -> str:
    """Post-process response to make citations clickable (XSS-safe)."""
    for s in sources:
        safe_url = html.escape(s.url)
        response = response.replace(
            f"[{s.citation_id}]",
            f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">[{s.citation_id}]</a>',
        )
    return response

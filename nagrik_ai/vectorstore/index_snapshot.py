from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nagrik_ai.crawler.crawl_manifest import CrawlManifest
from nagrik_ai.vectorstore.chroma_store import ChromaStore


class IndexedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    storage_id: str
    chunk_id: str
    document_id: str
    version_id: str
    retrieval_scopes: list[str] = Field(default_factory=list)


class VectorIndexSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    total_entries: int
    semantic_fingerprint: str
    duplicate_chunk_ids: list[str]
    entries: list[IndexedChunk]


def capture_index_snapshot(store: ChromaStore) -> VectorIndexSnapshot:
    entries: list[IndexedChunk] = []
    for document in store.get_all_documents():
        metadata: dict[str, Any] = document.metadata
        chunk_id = str(metadata.get("chunk_id", ""))
        raw_scopes = metadata.get("retrieval_scopes", "[]")
        if isinstance(raw_scopes, str):
            try:
                decoded = json.loads(raw_scopes)
                scopes = [str(scope) for scope in decoded] if isinstance(decoded, list) else [raw_scopes]
            except json.JSONDecodeError:
                scopes = [raw_scopes] if raw_scopes else []
        elif isinstance(raw_scopes, list):
            scopes = [str(scope) for scope in raw_scopes]
        else:
            scopes = []
        entries.append(
            IndexedChunk(
                storage_id=str(metadata.get("chroma_id", "")),
                chunk_id=chunk_id,
                document_id=str(metadata.get("document_id", "")),
                version_id=str(metadata.get("version_id", "")),
                retrieval_scopes=sorted(set(scopes)),
            )
        )
    entries.sort(key=lambda entry: (entry.document_id, entry.version_id, entry.chunk_id, entry.storage_id))
    counts = Counter(entry.chunk_id for entry in entries if entry.chunk_id)
    duplicates = sorted(chunk_id for chunk_id, count in counts.items() if count > 1)
    semantic = json.dumps([entry.model_dump(mode="json") for entry in entries], sort_keys=True, separators=(",", ":"))
    return VectorIndexSnapshot(
        generated_at=datetime.now(UTC),
        total_entries=len(entries),
        semantic_fingerprint=hashlib.sha256(semantic.encode()).hexdigest(),
        duplicate_chunk_ids=duplicates,
        entries=entries,
    )


def write_index_snapshot(snapshot: VectorIndexSnapshot, path: Path) -> None:
    payload = json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    CrawlManifest._atomic_write(path, payload.encode())

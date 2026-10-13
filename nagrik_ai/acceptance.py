from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from nagrik_ai.crawler.crawl_manifest import CrawlManifest
from nagrik_ai.crawler.url_utils import content_hash_for_bytes
from nagrik_ai.models.document import NormalizedDocument
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.utils.markdown_utils import read_markdown_file
from nagrik_ai.vectorstore.bm25_retriever import BM25Retriever
from nagrik_ai.vectorstore.index_snapshot import VectorIndexSnapshot
from nagrik_ai.vectorstore.legal_chunker import LegalChunker

ACCEPTANCE_SCHEMA_VERSION: Literal["1.0"] = "1.0"
MAX_CONTROLLED_PAGES = 50
MAX_CONTROLLED_DOCUMENTS = 100
ProbeKind = Literal["document_number", "hsn_sac", "amendment", "historical", "applicability"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CrawlLimits(StrictModel):
    page_cap: int = Field(ge=1, le=MAX_CONTROLLED_PAGES)
    document_cap: int = Field(ge=1, le=MAX_CONTROLLED_DOCUMENTS)
    depth: int = Field(default=0, ge=0, le=2)


class ManualReview(StrictModel):
    version_id: str
    reviewer: str
    reviewed_at: datetime
    metadata_matches_source: bool
    tables_match_source: bool | None = None
    notes: str = ""


class ReviewLedger(StrictModel):
    schema_version: Literal["1.0"] = ACCEPTANCE_SCHEMA_VERSION
    reviews: list[ManualReview] = Field(default_factory=list)


class ReviewSample(StrictModel):
    version_id: str
    document_id: str
    source_url: str
    normalized_path: str
    raw_path: str
    document_number: str | None
    hsn_sac_codes: list[str]
    page_count: int
    table_count: int
    table_row_count: int
    extraction_quality: str
    review_status: Literal["pending", "approved", "rejected"]
    review_notes: str = ""


class RetrievalProbe(StrictModel):
    kind: ProbeKind
    query: str
    scope: str
    expected_document_ids: list[str]
    returned_document_id: str | None = None
    citation_url: str | None = None
    page: int | None = None
    passed: bool


class IndexAudit(StrictModel):
    pending: int = 0
    complete: int = 0
    quarantined: int = 0
    transitions_pending: int = 0
    transitions_complete: int = 0
    committed_operation_ids: list[str] = Field(default_factory=list)


class VectorIndexAudit(StrictModel):
    expected_chunks: int
    indexed_chunks: int
    missing_chunk_ids: list[str] = Field(default_factory=list)
    orphaned_chunk_ids: list[str] = Field(default_factory=list)
    scope_mismatch_chunk_ids: list[str] = Field(default_factory=list)
    duplicate_chunk_ids: list[str] = Field(default_factory=list)
    relevant_fingerprint: str


class AcceptanceSnapshot(StrictModel):
    source_count: int
    content_version_count: int
    logical_document_count: int
    normalized_document_count: int
    parsed_artifact_count: int
    chunk_count: int
    semantic_fingerprint: str


class AcceptanceReport(StrictModel):
    schema_version: Literal["1.0"] = ACCEPTANCE_SCHEMA_VERSION
    generated_at: datetime
    site: str
    limits: CrawlLimits
    manifest_status_counts: dict[str, int]
    migration_status_counts: dict[str, int]
    index_audit: IndexAudit | None = None
    vector_index_audit: VectorIndexAudit | None = None
    snapshot: AcceptanceSnapshot
    review_samples: list[ReviewSample]
    retrieval_probes: list[RetrievalProbe]
    validation_errors: list[str]
    warnings: list[str]
    rollout_ready: bool


class _StaticDocumentSource:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def get_all_documents(self) -> list[Document]:
        return self.documents


def build_acceptance_report(
    site_root: Path,
    limits: CrawlLimits,
    *,
    sample_size: int = 5,
    review_ledger_path: Path | None = None,
    index_db_path: Path | None = None,
    vector_snapshot_path: Path | None = None,
    require_rollout_evidence: bool = False,
    retrieval_service: DocumentRetrievalService | None = None,
) -> AcceptanceReport:
    """Audit a bounded crawl without mutating its source or parsed artifacts."""
    root = site_root.resolve()
    manifest = CrawlManifest(root)
    errors: list[str] = []
    warnings: list[str] = []
    status_counts = Counter(outcome.status for outcome in manifest.state.outcomes)
    for status in ("success", "unchanged", "reused", "blocked", "incomplete", "failed"):
        status_counts.setdefault(status, 0)

    _audit_raw_artifacts(manifest, errors)
    documents, record_paths = _load_normalized_documents(root, errors)
    parsed_hashes = _parsed_artifact_hashes(root, documents, errors)
    chunks: list[Document] = []
    for document in documents:
        matches = sorted((root / "parsed" / "documents").glob(f"*/{document.version_id}.md"))
        if matches:
            metadata, content = read_markdown_file(str(matches[0]))
            chunks.extend(LegalChunker().chunk_markdown(content, metadata))
        else:
            chunks.extend(LegalChunker().chunk_document(document))
    _audit_chunks(chunks, errors)

    migration_counts = Counter(migration.state for migration in manifest.state.migrations.values())
    for state in ("pending", "file_moved", "committed", "quarantined"):
        migration_counts.setdefault(state, 0)
    if migration_counts["pending"] or migration_counts["file_moved"] or migration_counts["quarantined"]:
        errors.append("manifest contains unfinished or quarantined identity migrations")

    index_audit = _audit_index(index_db_path, errors) if index_db_path is not None else None
    vector_index_audit = (
        _audit_vector_index(vector_snapshot_path, documents, chunks, errors)
        if vector_snapshot_path is not None
        else None
    )
    ledger = _load_review_ledger(review_ledger_path, errors)
    samples = _select_review_samples(root, documents, record_paths, ledger, sample_size)
    if samples and any(sample.review_status != "approved" for sample in samples):
        errors.append("manual source comparison is incomplete or rejected for one or more review samples")

    probes = _retrieval_probes(documents, chunks, retrieval_service)
    if documents and not probes:
        warnings.append("controlled corpus has no supported identifiers for deterministic retrieval probes")
    if any(not probe.passed for probe in probes):
        errors.append("one or more deterministic retrieval probes failed")
    if status_counts["blocked"] or status_counts["failed"]:
        warnings.append("crawl recorded blocked or failed requests; inspect manifest outcomes before expansion")
    if status_counts["incomplete"]:
        errors.append("crawl contains incomplete responses")
    if not documents:
        errors.append("no normalized documents were available for acceptance review")
    if require_rollout_evidence:
        run_path = root / "manifests" / "last_crawl_run.json"
        if not run_path.is_file():
            errors.append("recorded bounded crawl settings are missing")
        else:
            try:
                run = json.loads(run_path.read_text(encoding="utf-8"))
                for field in ("page_cap", "document_cap", "depth"):
                    if run.get(field) != getattr(limits, field):
                        errors.append(f"audit {field} does not match recorded crawl settings")
            except (ValueError, OSError) as exc:
                errors.append(f"invalid crawl run provenance: {exc}")
        if vector_snapshot_path is None or index_db_path is None:
            errors.append("rollout requires both the vector snapshot and transactional index database")
        if retrieval_service is None:
            errors.append("rollout requires retrieval probes against the persisted hybrid index")
        if not any(document.tables for document in documents):
            errors.append("rollout corpus has no structured rate table to review")
        if len(parsed_hashes) != len(documents):
            errors.append("not every normalized document has a parsed artifact")

    snapshot = _snapshot(manifest, documents, parsed_hashes, chunks, index_audit, vector_index_audit)
    return AcceptanceReport(
        generated_at=datetime.now(UTC),
        site=manifest.state.site,
        limits=limits,
        manifest_status_counts=dict(sorted(status_counts.items())),
        migration_status_counts=dict(sorted(migration_counts.items())),
        index_audit=index_audit,
        vector_index_audit=vector_index_audit,
        snapshot=snapshot,
        review_samples=samples,
        retrieval_probes=probes,
        validation_errors=errors,
        warnings=warnings,
        rollout_ready=not errors,
    )


def write_acceptance_report(report: AcceptanceReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    CrawlManifest._atomic_write(path, payload.encode())


def compare_acceptance_reports(first_path: Path, second_path: Path) -> list[str]:
    first = AcceptanceReport.model_validate_json(first_path.read_text(encoding="utf-8"))
    second = AcceptanceReport.model_validate_json(second_path.read_text(encoding="utf-8"))
    differences: list[str] = []
    for field in (
        "source_count",
        "content_version_count",
        "logical_document_count",
        "normalized_document_count",
        "parsed_artifact_count",
        "chunk_count",
        "semantic_fingerprint",
    ):
        if getattr(first.snapshot, field) != getattr(second.snapshot, field):
            differences.append(field)
    return differences


def _audit_raw_artifacts(manifest: CrawlManifest, errors: list[str]) -> None:
    for version_id, version in sorted(manifest.state.versions.items()):
        path = manifest.resolve_path(version.local_path)
        if not path.is_file():
            errors.append(f"raw artifact missing for version {version_id}: {version.local_path}")
            continue
        if content_hash_for_bytes(path.read_bytes()) != version.content_hash:
            errors.append(f"raw artifact hash mismatch for version {version_id}: {version.local_path}")


def _load_normalized_documents(
    root: Path,
    errors: list[str],
) -> tuple[list[NormalizedDocument], dict[str, Path]]:
    documents: list[NormalizedDocument] = []
    record_paths: dict[str, Path] = {}
    records_root = root / "parsed" / "records"
    for path in sorted(records_root.rglob("*.json")) if records_root.exists() else []:
        try:
            document = NormalizedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid normalized record {path.relative_to(root)}: {exc}")
            continue
        documents.append(document)
        record_paths[document.version_id] = path
    return documents, record_paths


def _parsed_artifact_hashes(
    root: Path,
    documents: list[NormalizedDocument],
    errors: list[str],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    parsed_root = root / "parsed" / "documents"
    for document in documents:
        matches = sorted(parsed_root.glob(f"*/{document.version_id}.md")) if parsed_root.exists() else []
        if not matches:
            continue
        if len(matches) > 1:
            errors.append(f"multiple parsed artifacts found for version {document.version_id}")
            continue
        hashes[document.version_id] = content_hash_for_bytes(matches[0].read_bytes())
    return hashes


def _audit_chunks(chunks: list[Document], errors: list[str]) -> None:
    ids = [str(chunk.metadata.get("chunk_id", "")) for chunk in chunks]
    if "" in ids:
        errors.append("one or more legal chunks have no stable chunk ID")
    duplicates = [chunk_id for chunk_id, count in Counter(ids).items() if chunk_id and count > 1]
    if duplicates:
        errors.append(f"duplicate stable chunk IDs: {', '.join(sorted(duplicates))}")
    for chunk in chunks:
        if not chunk.metadata.get("citation_url"):
            errors.append(f"chunk {chunk.metadata.get('chunk_id', '<unknown>')} has no citation URL")
        if chunk.metadata.get("source_format") == "pdf" and int(chunk.metadata.get("page", 0)) < 1:
            errors.append(f"PDF chunk {chunk.metadata.get('chunk_id', '<unknown>')} has no page citation")


def _load_review_ledger(path: Path | None, errors: list[str]) -> dict[str, ManualReview]:
    if path is None:
        return {}
    try:
        ledger = ReviewLedger.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid manual review ledger: {exc}")
        return {}
    return {review.version_id: review for review in ledger.reviews}


def _select_review_samples(
    root: Path,
    documents: list[NormalizedDocument],
    record_paths: dict[str, Path],
    ledger: dict[str, ManualReview],
    sample_size: int,
) -> list[ReviewSample]:
    ordered = sorted(documents, key=lambda document: (-len(document.tables), document.document_id, document.version_id))
    samples: list[ReviewSample] = []
    for document in ordered[: max(sample_size, 0)]:
        review = ledger.get(document.version_id)
        approved = bool(
            review
            and review.metadata_matches_source
            and (not document.tables or review.tables_match_source is True)
        )
        rejected = bool(
            review
            and (
                not review.metadata_matches_source
                or (bool(document.tables) and review.tables_match_source is not True)
            )
        )
        samples.append(
            ReviewSample(
                version_id=document.version_id,
                document_id=document.document_id,
                source_url=document.canonical_url,
                normalized_path=record_paths[document.version_id].relative_to(root).as_posix(),
                raw_path=document.raw_local_path,
                document_number=document.legal_metadata.document_number,
                hsn_sac_codes=document.legal_metadata.hsn_sac_codes,
                page_count=len(document.pages),
                table_count=len(document.tables),
                table_row_count=sum(len(table.rows) for table in document.tables),
                extraction_quality=document.extraction.quality_status,
                review_status="approved" if approved else ("rejected" if rejected else "pending"),
                review_notes=review.notes if review else "",
            )
        )
    return samples


def _retrieval_probes(
    documents: list[NormalizedDocument],
    chunks: list[Document],
    retrieval_service: DocumentRetrievalService | None = None,
) -> list[RetrievalProbe]:
    if not chunks:
        return []
    retriever = BM25Retriever(_StaticDocumentSource(chunks))
    identifiers: dict[tuple[ProbeKind, str, str], set[str]] = defaultdict(set)
    for document in documents:
        metadata = document.legal_metadata
        if metadata.document_number:
            identifiers[("document_number", metadata.document_number, "all_versions")].add(document.document_id)
            if document.source_version_status == "historical_source":
                identifiers[("historical", metadata.document_number, "historical_source")].add(document.document_id)
        for code in metadata.hsn_sac_codes:
            identifiers[("hsn_sac", code, "all_versions")].add(document.document_id)
        table_codes: set[str] = set()
        for table in document.tables:
            for row in table.rows:
                for key, value in row.cells.items():
                    if any(term in key.casefold() for term in ("chapter", "heading", "hsn", "sac")):
                        table_codes.update(re.findall(r"\b\d{4}(?:\s\d{2}){0,2}\b", value))
        for code in sorted(table_codes)[:5]:
            identifiers[("hsn_sac", code, "all_versions")].add(document.document_id)
        for amendment_id in metadata.amendment_document_ids:
            identifiers[("amendment", amendment_id, "all_versions")].add(document.document_id)
        if metadata.applicability.status == "applicable" and metadata.document_number:
            identifiers[("applicability", metadata.document_number, "legally_applicable")].add(document.document_id)

    probes: list[RetrievalProbe] = []
    for (kind, query, scope), expected in sorted(identifiers.items())[:50]:
        result = (
            retrieval_service.retrieve(query, scope=scope)
            if retrieval_service is not None
            else retriever.retrieve(query, k=1, scope=scope)
        )
        top = result[0] if result else None
        metadata = top.get("metadata", {}) if top else {}
        returned = str(metadata.get("document_id", "")) or None
        citation_url = str(metadata.get("citation_url", "")) or None
        page = int(metadata.get("page", 0)) or None
        citation_valid = citation_url is not None
        if metadata.get("source_format") == "pdf":
            citation_valid = citation_valid and page is not None
        probes.append(
            RetrievalProbe(
                kind=kind,
                query=query,
                scope=scope,
                expected_document_ids=sorted(expected),
                returned_document_id=returned,
                citation_url=citation_url,
                page=page,
                passed=returned in expected and citation_valid,
            )
        )
    return probes


def _audit_index(path: Path, errors: list[str]) -> IndexAudit | None:
    if not path.is_file():
        errors.append(f"index state database does not exist: {path}")
        return None
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        with connection:
            outbox = Counter(
                {row["state"]: row["count"] for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM index_outbox GROUP BY state"
                )}
            )
            transitions = Counter(
                {row["transition_state"]: row["count"] for row in connection.execute(
                    "SELECT transition_state, COUNT(*) AS count FROM index_state GROUP BY transition_state"
                )}
            )
            committed_ids = [
                row["desired_operation_id"]
                for row in connection.execute(
                    "SELECT desired_operation_id FROM index_state WHERE transition_state = 'complete' ORDER BY 1"
                )
            ]
    except (sqlite3.Error, OSError) as exc:
        errors.append(f"could not audit index state database: {exc}")
        return None
    finally:
        if connection is not None:
            connection.close()
    audit = IndexAudit(
        pending=outbox["pending"],
        complete=outbox["complete"],
        quarantined=outbox["quarantined"],
        transitions_pending=transitions["pending"],
        transitions_complete=transitions["complete"],
        committed_operation_ids=committed_ids,
    )
    if audit.pending or audit.quarantined or audit.transitions_pending:
        errors.append("retrieval index has pending or quarantined operations")
    return audit


def _audit_vector_index(
    path: Path,
    documents: list[NormalizedDocument],
    chunks: list[Document],
    errors: list[str],
) -> VectorIndexAudit | None:
    if not path.is_file():
        errors.append(f"vector index snapshot does not exist: {path}")
        return None
    try:
        snapshot = VectorIndexSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid vector index snapshot: {exc}")
        return None
    expected = {
        str(chunk.metadata["chunk_id"]): set(str(scope) for scope in chunk.metadata.get("retrieval_scopes", []))
        for chunk in chunks
    }
    document_ids = {document.document_id for document in documents}
    relevant_entries = [entry for entry in snapshot.entries if entry.document_id in document_ids]
    indexed = {entry.chunk_id: set(entry.retrieval_scopes) for entry in relevant_entries if entry.chunk_id}
    missing = sorted(set(expected) - set(indexed))
    orphaned = sorted(set(indexed) - set(expected))
    scope_mismatches = sorted(
        chunk_id for chunk_id in set(expected) & set(indexed) if expected[chunk_id] != indexed[chunk_id]
    )
    unstable_ids = sorted(
        entry.chunk_id
        for entry in relevant_entries
        if entry.chunk_id and entry.storage_id and entry.storage_id != entry.chunk_id
    )
    duplicates = sorted(set(snapshot.duplicate_chunk_ids) & set(indexed))
    if missing:
        errors.append("vector index is missing expected chunks")
    if orphaned:
        errors.append("vector index contains orphaned chunks for the controlled corpus")
    if scope_mismatches:
        errors.append("vector index retrieval scopes differ from the controlled corpus")
    if duplicates:
        errors.append("vector index contains duplicate stable chunk IDs")
    if unstable_ids:
        errors.append("vector index storage IDs do not match stable legal chunk IDs")
    relevant_payload = json.dumps(
        [entry.model_dump(mode="json") for entry in relevant_entries],
        sort_keys=True,
        separators=(",", ":"),
    )
    return VectorIndexAudit(
        expected_chunks=len(expected),
        indexed_chunks=len(indexed),
        missing_chunk_ids=missing,
        orphaned_chunk_ids=orphaned,
        scope_mismatch_chunk_ids=scope_mismatches,
        duplicate_chunk_ids=duplicates,
        relevant_fingerprint=hashlib.sha256(relevant_payload.encode()).hexdigest(),
    )


def _snapshot(
    manifest: CrawlManifest,
    documents: list[NormalizedDocument],
    parsed_hashes: dict[str, str],
    chunks: list[Document],
    index_audit: IndexAudit | None,
    vector_index_audit: VectorIndexAudit | None,
) -> AcceptanceSnapshot:
    logical_ids = {
        document_id
        for document_id in (source.document_id for source in manifest.state.sources.values())
        if document_id is not None
    } | {document.document_id for document in documents}
    semantic = {
        "sources": {
            source_id: {
                "canonical_url": source.canonical_url,
                "document_id": source.document_id,
                "current_version_id": source.current_version_id,
            }
            for source_id, source in sorted(manifest.state.sources.items())
        },
        "versions": {
            version_id: {
                "source_id": version.source_id,
                "document_id": version.document_id,
                "content_hash": version.content_hash,
                "previous_version_id": version.previous_version_id,
            }
            for version_id, version in sorted(manifest.state.versions.items())
        },
        "normalized": {
            document.version_id: {
                "document_id": document.document_id,
                "content_hash": document.content_hash,
                "source_version_status": document.source_version_status,
                "legal_metadata": document.legal_metadata.model_dump(mode="json"),
                "pages": [page.model_dump(mode="json") for page in document.pages],
                "tables": [table.model_dump(mode="json") for table in document.tables],
            }
            for document in sorted(documents, key=lambda item: item.version_id)
        },
        "parsed_hashes": dict(sorted(parsed_hashes.items())),
        "chunks": [
            {
                "chunk_id": str(chunk.metadata.get("chunk_id", "")),
                "content_hash": content_hash_for_bytes(chunk.page_content.encode()),
                "retrieval_scopes": chunk.metadata.get("retrieval_scopes", []),
            }
            for chunk in sorted(chunks, key=lambda item: str(item.metadata.get("chunk_id", "")))
        ],
        "committed_index_operations": index_audit.committed_operation_ids if index_audit else [],
        "vector_index_fingerprint": vector_index_audit.relevant_fingerprint if vector_index_audit else None,
    }
    canonical = json.dumps(semantic, sort_keys=True, separators=(",", ":"), default=str)
    return AcceptanceSnapshot(
        source_count=len(manifest.state.sources),
        content_version_count=len(manifest.state.versions),
        logical_document_count=len(logical_ids),
        normalized_document_count=len(documents),
        parsed_artifact_count=len(parsed_hashes),
        chunk_count=len(chunks),
        semantic_fingerprint=hashlib.sha256(canonical.encode()).hexdigest(),
    )

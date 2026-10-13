from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nagrik_ai.vectorstore.legal_chunker import RetrievalScope

IndexAction = Literal["none", "parse", "chunk", "embed", "index"]
OperationState = Literal["pending", "complete", "quarantined"]


class IndexFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content_hash: str
    schema_version: str
    parser_name: str
    parser_version: str
    extraction_parser_name: str = ""
    extraction_parser_version: str = ""
    chunker_name: str
    chunker_version: str
    embedding_provider: str
    embedding_model: str
    embedding_model_version: str


class ReprocessingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: tuple[IndexAction, ...]
    reasons: tuple[str, ...]


class IndexOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str
    document_id: str
    target_version_id: str
    desired_scopes: list[RetrievalScope]
    expected_chunk_ids: list[str]
    fingerprint: IndexFingerprint
    state: OperationState = "pending"
    attempts: int = 0
    last_error: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def reprocessing_plan(current: IndexFingerprint | None, desired: IndexFingerprint) -> ReprocessingPlan:
    if current is None:
        return ReprocessingPlan(actions=("parse", "chunk", "embed", "index"), reasons=("not_indexed",))
    actions: list[IndexAction] = []
    reasons: list[str] = []
    if current.content_hash != desired.content_hash:
        actions.extend(("parse", "chunk", "embed", "index"))
        reasons.append("source_bytes_changed")
    if (
        current.schema_version != desired.schema_version
        or current.parser_name != desired.parser_name
        or current.parser_version != desired.parser_version
        or current.extraction_parser_name != desired.extraction_parser_name
        or current.extraction_parser_version != desired.extraction_parser_version
    ):
        actions.extend(("parse", "chunk", "embed", "index"))
        reasons.append("parser_or_schema_changed")
    if current.chunker_name != desired.chunker_name or current.chunker_version != desired.chunker_version:
        actions.extend(("chunk", "embed", "index"))
        reasons.append("chunker_changed")
    if (
        current.embedding_provider != desired.embedding_provider
        or current.embedding_model != desired.embedding_model
        or current.embedding_model_version != desired.embedding_model_version
    ):
        actions.extend(("embed", "index"))
        reasons.append("embedding_model_changed")
    ordered = tuple(action for action in ("parse", "chunk", "embed", "index") if action in actions)
    return ReprocessingPlan(actions=ordered or ("none",), reasons=tuple(reasons))


class IndexOutbox:
    """SQLite source of truth for desired and last-verified retrieval state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS index_state (
                    state_key TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    target_version_id TEXT NOT NULL,
                    desired_operation_id TEXT NOT NULL,
                    desired_payload TEXT NOT NULL,
                    committed_payload TEXT,
                    transition_state TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS index_outbox (
                    operation_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS index_chunk_payloads (
                    operation_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS legal_relationships (
                    relationship_id TEXT PRIMARY KEY,
                    source_document_id TEXT NOT NULL,
                    target_document_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    provenance_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS applicability_facts (
                    assertion_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    subject TEXT,
                    provenance_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rate_records (
                    rate_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    hsn_sac TEXT NOT NULL,
                    description TEXT NOT NULL,
                    rate_json TEXT NOT NULL,
                    conditions_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL
                );
                """
            )

    def plan(
        self,
        *,
        document_id: str,
        target_version_id: str,
        desired_scopes: list[RetrievalScope],
        expected_chunk_ids: list[str],
        fingerprint: IndexFingerprint,
        chunk_payloads: list[dict[str, object]] | None = None,
    ) -> IndexOperation | None:
        normalized_scopes: list[RetrievalScope] = sorted(set(desired_scopes))
        normalized_chunk_ids = sorted(set(expected_chunk_ids))
        payload_without_id = {
            "document_id": document_id,
            "target_version_id": target_version_id,
            "desired_scopes": normalized_scopes,
            "expected_chunk_ids": normalized_chunk_ids,
            "fingerprint": fingerprint.model_dump(mode="json"),
        }
        canonical = json.dumps(payload_without_id, sort_keys=True, separators=(",", ":"))
        operation_id = hashlib.sha256(canonical.encode()).hexdigest()
        operation = IndexOperation(
            operation_id=operation_id,
            document_id=document_id,
            target_version_id=target_version_id,
            desired_scopes=normalized_scopes,
            expected_chunk_ids=normalized_chunk_ids,
            fingerprint=fingerprint,
        )
        payload = operation.model_dump_json()
        state_key = hashlib.sha256(f"{document_id}\n{target_version_id}".encode()).hexdigest()
        with self._connect() as connection:
            if chunk_payloads is not None:
                connection.execute(
                    "INSERT OR REPLACE INTO index_chunk_payloads VALUES (?, ?)",
                    (operation_id, json.dumps(chunk_payloads, sort_keys=True)),
                )
                self._record_legal_payloads(connection, document_id, target_version_id, chunk_payloads)
            row = connection.execute(
                "SELECT committed_payload, desired_operation_id FROM index_state WHERE state_key = ?",
                (state_key,),
            ).fetchone()
            if row is not None and row["committed_payload"] is not None:
                committed = IndexOperation.model_validate_json(row["committed_payload"])
                if committed.operation_id == operation_id:
                    return None
            connection.execute(
                """
                INSERT INTO index_outbox(operation_id, document_id, payload, state, attempts, created_at)
                VALUES (?, ?, ?, 'pending', 0, ?)
                ON CONFLICT(operation_id) DO UPDATE SET
                    state = CASE WHEN index_outbox.state = 'complete' THEN 'complete' ELSE 'pending' END,
                    last_error = NULL
                """,
                (operation_id, document_id, payload, operation.created_at),
            )
            connection.execute(
                """
                INSERT INTO index_state(
                    state_key, document_id, target_version_id, desired_operation_id,
                    desired_payload, committed_payload, transition_state
                )
                VALUES (?, ?, ?, ?, ?, NULL, 'pending')
                ON CONFLICT(state_key) DO UPDATE SET
                    desired_operation_id = excluded.desired_operation_id,
                    desired_payload = excluded.desired_payload,
                    transition_state = CASE
                        WHEN index_state.committed_payload = excluded.desired_payload THEN 'complete'
                        ELSE 'pending'
                    END
                """,
                (state_key, document_id, target_version_id, operation_id, payload),
            )
        return operation

    def chunk_payloads(self, operation_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM index_chunk_payloads WHERE operation_id = ?", (operation_id,)
            ).fetchone()
        if row is None:
            raise ValueError("No durable chunk payload for index operation")
        payload: list[dict[str, object]] = json.loads(row["payload"])
        return payload

    @staticmethod
    def _record_legal_payloads(
        connection: sqlite3.Connection,
        document_id: str,
        version_id: str,
        chunks: list[dict[str, object]],
    ) -> None:
        connection.execute(
            "DELETE FROM rate_records WHERE document_id = ? AND json_extract(provenance_json, '$.version_id') = ?",
            (document_id, version_id),
        )
        connection.execute(
            "DELETE FROM legal_relationships WHERE source_document_id = ? "
            "AND json_extract(provenance_json, '$.version_id') = ?",
            (document_id, version_id),
        )
        for chunk in chunks:
            metadata = chunk.get("metadata")
            if not isinstance(metadata, dict):
                continue
            raw_legal = metadata.get("legal_metadata", {})
            legal = json.loads(raw_legal) if isinstance(raw_legal, str) else raw_legal
            if isinstance(legal, dict):
                applicability = legal.get("applicability", {})
                if isinstance(applicability, dict):
                    assertion_id = hashlib.sha256(f"{document_id}\n{version_id}\napplicability".encode()).hexdigest()
                    start = applicability.get("effective_date", {})
                    end = applicability.get("rescission_or_supersession_date", {})
                    connection.execute(
                        "INSERT OR REPLACE INTO applicability_facts VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            assertion_id, document_id, applicability.get("status", "unknown"),
                            start.get("value") if isinstance(start, dict) else None,
                            end.get("value") if isinstance(end, dict) else None,
                            legal.get("subject"), json.dumps({"version_id": version_id, "facts": applicability}),
                        ),
                    )
                for target in legal.get("amendment_document_ids", []):
                    relationship_id = hashlib.sha256(f"{document_id}\n{version_id}\n{target}".encode()).hexdigest()
                    connection.execute(
                        "INSERT OR REPLACE INTO legal_relationships VALUES (?, ?, ?, ?, ?)",
                        (relationship_id, document_id, target, "amendment", json.dumps(metadata, sort_keys=True)),
                    )
            raw_row = metadata.get("rate_row", {})
            row = json.loads(raw_row) if isinstance(raw_row, str) else raw_row
            if isinstance(row, dict) and isinstance(row.get("cells"), dict):
                cells = row["cells"]
                code = next((str(value) for key, value in cells.items() if "hsn" in key.lower()
                             or "sac" in key.lower() or "chapter" in key.lower()), "")
                rates = {key: value for key, value in cells.items() if "rate" in key.lower()
                         or "tax" in key.lower() or "cess" in key.lower()}
                connection.execute(
                    "INSERT OR REPLACE INTO rate_records VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(metadata.get("chunk_id")), document_id, code,
                        json.dumps(cells, sort_keys=True), json.dumps(rates, sort_keys=True),
                        json.dumps([row.get("condition"), *row.get("footnotes", [])]),
                        json.dumps({"version_id": version_id, "source_url": metadata.get("citation_url"), "row": row}),
                    ),
                )

    def pending(self) -> list[IndexOperation]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload, attempts, last_error FROM index_outbox WHERE state = 'pending' ORDER BY created_at"
            ).fetchall()
        operations: list[IndexOperation] = []
        for row in rows:
            operation = IndexOperation.model_validate_json(row["payload"])
            operations.append(
                operation.model_copy(update={"attempts": row["attempts"], "last_error": row["last_error"]})
            )
        return operations

    def desired_operations(self) -> list[IndexOperation]:
        with self._connect() as connection:
            rows = connection.execute("SELECT desired_payload FROM index_state ORDER BY document_id").fetchall()
        return [IndexOperation.model_validate_json(row["desired_payload"]) for row in rows]

    def mark_complete(self, operation: IndexOperation, observed: dict[str, set[str]]) -> None:
        expected_ids = set(operation.expected_chunk_ids)
        if set(observed) != expected_ids:
            raise ValueError("observed chunk IDs do not match the desired index state")
        desired_scopes = set(operation.desired_scopes)
        if any(scopes != desired_scopes for scopes in observed.values()):
            raise ValueError("observed retrieval scopes do not match the desired index state")
        payload = operation.model_copy(
            update={"state": "complete", "attempts": operation.attempts, "last_error": None}
        ).model_dump_json()
        with self._connect() as connection:
            desired = connection.execute(
                """
                SELECT desired_operation_id FROM index_state
                WHERE document_id = ? AND target_version_id = ?
                """,
                (operation.document_id, operation.target_version_id),
            ).fetchone()
            if desired is None or desired["desired_operation_id"] != operation.operation_id:
                raise ValueError("operation is no longer the desired state")
            connection.execute(
                "UPDATE index_outbox SET state = 'complete', last_error = NULL WHERE operation_id = ?",
                (operation.operation_id,),
            )
            connection.execute(
                """
                UPDATE index_state SET committed_payload = ?, transition_state = 'complete'
                WHERE document_id = ? AND target_version_id = ?
                """,
                (payload, operation.document_id, operation.target_version_id),
            )

    def mark_retry(self, operation_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE index_outbox
                SET attempts = attempts + 1, last_error = ?, state = 'pending'
                WHERE operation_id = ?
                """,
                (error, operation_id),
            )

    def quarantine(self, operation_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE index_outbox SET state = 'quarantined', last_error = ? WHERE operation_id = ?",
                (error, operation_id),
            )

    def committed(self, document_id: str, target_version_id: str | None = None) -> IndexOperation | None:
        with self._connect() as connection:
            if target_version_id is None:
                rows = connection.execute(
                    "SELECT committed_payload FROM index_state WHERE document_id = ?", (document_id,)
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT committed_payload FROM index_state
                    WHERE document_id = ? AND target_version_id = ?
                    """,
                    (document_id, target_version_id),
                ).fetchall()
        operations = [
            IndexOperation.model_validate_json(row["committed_payload"])
            for row in rows
            if row["committed_payload"] is not None
        ]
        if not operations:
            return None
        return next(
            (operation for operation in operations if "current_source" in operation.desired_scopes),
            operations[-1],
        )

    def record_relationship(
        self,
        source_document_id: str,
        target_document_id: str,
        relationship_type: str,
        provenance: dict[str, object],
    ) -> str:
        identity = f"{source_document_id}\n{target_document_id}\n{relationship_type}"
        relationship_id = hashlib.sha256(identity.encode()).hexdigest()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO legal_relationships VALUES (?, ?, ?, ?, ?)",
                (
                    relationship_id,
                    source_document_id,
                    target_document_id,
                    relationship_type,
                    json.dumps(provenance, sort_keys=True),
                ),
            )
        return relationship_id

    def record_applicability(
        self,
        assertion_id: str,
        document_id: str,
        status: str,
        provenance: dict[str, object],
        *,
        valid_from: str | None = None,
        valid_to: str | None = None,
        subject: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO applicability_facts VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    assertion_id,
                    document_id,
                    status,
                    valid_from,
                    valid_to,
                    subject,
                    json.dumps(provenance, sort_keys=True),
                ),
            )

    def record_rate(
        self,
        rate_id: str,
        document_id: str,
        hsn_sac: str,
        description: str,
        rates: dict[str, str],
        conditions: list[str],
        provenance: dict[str, object],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO rate_records VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    rate_id,
                    document_id,
                    hsn_sac,
                    description,
                    json.dumps(rates, sort_keys=True),
                    json.dumps(conditions, sort_keys=True),
                    json.dumps(provenance, sort_keys=True),
                ),
            )

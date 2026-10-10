from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from nagrik_ai.crawler.url_utils import (
    canonicalize_url,
    content_hash_for_bytes,
    source_id_for_url,
    version_id_for_content,
)

MANIFEST_SCHEMA_VERSION = 1

FetchStatus = Literal["success", "unchanged", "reused", "blocked", "failed", "incomplete"]
IdentityStatus = Literal["unresolved", "resolved"]
MigrationState = Literal["pending", "file_moved", "committed", "quarantined"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SourceRecord(BaseModel):
    source_id: str
    canonical_url: str
    observed_urls: list[str] = Field(default_factory=list)
    site: str
    document_id: str | None = None
    current_version_id: str | None = None
    identity_status: IdentityStatus = "unresolved"
    identity_resolution_method: str = "source_url_provisional"
    identity_confidence: float | None = None
    last_fetch_time: str | None = None
    last_status: FetchStatus | None = None


class ManifestRecord(BaseModel):
    source_id: str
    document_id: str | None = None
    version_id: str
    canonical_source_url: str
    observed_source_url: str
    site: str
    content_type: str
    local_path: str
    content_hash: str
    fetch_time: str
    status: FetchStatus
    previous_version_id: str | None = None
    identity_resolution_method: str = "source_url_provisional"
    identity_status: IdentityStatus = "unresolved"
    identity_confidence: float | None = None
    title: str = ""
    parsed_path: str | None = None


class FetchOutcome(BaseModel):
    source_id: str
    canonical_source_url: str
    observed_source_url: str
    site: str
    fetch_time: str
    status: FetchStatus
    http_status: int | None = None
    error: str | None = None


class MigrationRecord(BaseModel):
    migration_id: str
    source_id: str
    version_id: str
    document_id: str
    source_path: str
    destination_path: str
    expected_content_hash: str
    resolution_evidence: dict[str, object] = Field(default_factory=dict)
    identity_resolution_method: str
    identity_confidence: float | None = None
    state: MigrationState = "pending"
    former_path: str | None = None
    resolution_timestamp: str | None = None
    attempts: int = 0
    last_error: str | None = None


class ManifestState(BaseModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    site: str
    sources: dict[str, SourceRecord] = Field(default_factory=dict)
    versions: dict[str, ManifestRecord] = Field(default_factory=dict)
    outcomes: list[FetchOutcome] = Field(default_factory=list)
    migrations: dict[str, MigrationRecord] = Field(default_factory=dict)


class CrawlManifest:
    def __init__(self, site_root: Path, site: str | None = None) -> None:
        self.site_root = site_root.resolve()
        self.crawled_dir = self.site_root / "crawled"
        self.path = self.site_root / "manifests" / "crawl_state.json"
        self.site = site or self.site_root.name
        self.state = self._load()

    @classmethod
    def for_crawled_dir(cls, crawled_dir: Path, site: str | None = None) -> CrawlManifest:
        return cls(crawled_dir.resolve().parent, site=site)

    def _load(self) -> ManifestState:
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return ManifestState.model_validate(payload)
        state = ManifestState(site=self.site)
        if self._import_legacy_mappings(state):
            self.state = state
            self.save()
        return state

    def _import_legacy_mappings(self, state: ManifestState) -> bool:
        mapping_path = self.crawled_dir / "url_mappings.json"
        if not mapping_path.exists():
            return False
        try:
            raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(raw, dict):
            return False

        imported = False
        for filename, metadata in raw.items():
            if not isinstance(filename, str) or not isinstance(metadata, dict):
                continue
            url = metadata.get("url")
            if not isinstance(url, str) or not url:
                continue
            file_path = self.crawled_dir / filename
            if not file_path.is_file():
                continue
            content = file_path.read_bytes()
            content_hash = content_hash_for_bytes(content)
            source_id = source_id_for_url(url)
            version_id = version_id_for_content(source_id, content_hash)
            fetch_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC).isoformat()
            relative_path = file_path.resolve().relative_to(self.site_root).as_posix()
            canonical_url = canonicalize_url(url)
            source = SourceRecord(
                source_id=source_id,
                canonical_url=canonical_url,
                observed_urls=[url],
                site=self.site,
                current_version_id=version_id,
                last_fetch_time=fetch_time,
                last_status="success",
            )
            state.sources[source_id] = source
            state.versions[version_id] = ManifestRecord(
                source_id=source_id,
                version_id=version_id,
                canonical_source_url=canonical_url,
                observed_source_url=url,
                site=self.site,
                content_type="text/html",
                local_path=relative_path,
                content_hash=content_hash,
                fetch_time=fetch_time,
                status="success",
                title=str(metadata.get("title", "")),
            )
            imported = True
        return imported

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state.model_dump(mode="json")
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        self._atomic_write(self.path, serialized.encode("utf-8"))

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
                temporary_path = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def reusable_version(self, url: str) -> ManifestRecord | None:
        source = self.state.sources.get(source_id_for_url(url))
        if source is None or source.current_version_id is None:
            return None
        version = self.state.versions.get(source.current_version_id)
        if version is None:
            return None
        local_path = self.resolve_path(version.local_path)
        if not local_path.is_file():
            return None
        if content_hash_for_bytes(local_path.read_bytes()) != version.content_hash:
            return None
        return version

    def store_response(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str,
        title: str = "",
        fetch_time: str | None = None,
    ) -> tuple[ManifestRecord, bool]:
        observed_url = url
        canonical_url = canonicalize_url(url)
        source_id = source_id_for_url(url)
        content_hash = content_hash_for_bytes(content)
        version_id = version_id_for_content(source_id, content_hash)
        fetched_at = fetch_time or utc_now()
        source = self.state.sources.get(source_id)
        if source is None:
            source = SourceRecord(
                source_id=source_id,
                canonical_url=canonical_url,
                observed_urls=[observed_url],
                site=self.site,
            )
            self.state.sources[source_id] = source
        elif observed_url not in source.observed_urls:
            source.observed_urls.append(observed_url)

        existing = self.state.versions.get(version_id)
        created = existing is None
        if existing is None:
            extension = self._extension_for_content_type(content_type)
            relative_path = Path("crawled") / extension.lstrip(".") / source_id / f"{version_id}{extension}"
            destination = self.resolve_path(relative_path.as_posix())
            self._atomic_write(destination, content)
            existing = ManifestRecord(
                source_id=source_id,
                document_id=source.document_id,
                version_id=version_id,
                canonical_source_url=canonical_url,
                observed_source_url=observed_url,
                site=self.site,
                content_type=content_type,
                local_path=relative_path.as_posix(),
                content_hash=content_hash,
                fetch_time=fetched_at,
                status="success",
                previous_version_id=source.current_version_id,
                identity_resolution_method=source.identity_resolution_method,
                identity_status=source.identity_status,
                identity_confidence=source.identity_confidence,
                title=title,
            )
            self.state.versions[version_id] = existing
            status: FetchStatus = "success"
        else:
            status = "unchanged"
            destination = self.resolve_path(existing.local_path)
            if not destination.exists():
                self._atomic_write(destination, content)

        source.current_version_id = version_id
        source.last_fetch_time = fetched_at
        source.last_status = status
        self.state.outcomes.append(
            FetchOutcome(
                source_id=source_id,
                canonical_source_url=canonical_url,
                observed_source_url=observed_url,
                site=self.site,
                fetch_time=fetched_at,
                status=status,
            )
        )
        self.save()
        return existing, created

    @staticmethod
    def _extension_for_content_type(content_type: str) -> str:
        normalized = content_type.partition(";")[0].strip().lower()
        if normalized == "application/pdf":
            return ".pdf"
        if normalized in {"application/json", "application/ld+json"} or normalized.endswith("+json"):
            return ".json"
        return ".html"

    def record_outcome(
        self,
        *,
        url: str,
        status: FetchStatus,
        http_status: int | None = None,
        error: str | None = None,
        fetch_time: str | None = None,
    ) -> FetchOutcome:
        source_id = source_id_for_url(url)
        canonical_url = canonicalize_url(url)
        fetched_at = fetch_time or utc_now()
        source = self.state.sources.get(source_id)
        if source is None:
            source = SourceRecord(
                source_id=source_id,
                canonical_url=canonical_url,
                observed_urls=[url],
                site=self.site,
            )
            self.state.sources[source_id] = source
        elif url not in source.observed_urls:
            source.observed_urls.append(url)
        source.last_fetch_time = fetched_at
        source.last_status = status
        outcome = FetchOutcome(
            source_id=source_id,
            canonical_source_url=canonical_url,
            observed_source_url=url,
            site=self.site,
            fetch_time=fetched_at,
            status=status,
            http_status=http_status,
            error=error,
        )
        self.state.outcomes.append(outcome)
        self.save()
        return outcome

    def mark_reused(self, url: str) -> ManifestRecord | None:
        version = self.reusable_version(url)
        if version is None:
            return None
        self.record_outcome(url=url, status="reused")
        return version

    def register_parsed_artifact(self, version_id: str, path: Path) -> None:
        version = self.state.versions[version_id]
        version.parsed_path = path.resolve().relative_to(self.site_root).as_posix()
        self.save()

    def resolve_document_identity(
        self,
        *,
        source_id: str,
        document_id: str,
        method: str,
        evidence: dict[str, object],
        confidence: float | None = None,
    ) -> list[MigrationRecord]:
        source = self.state.sources[source_id]
        migrations: list[MigrationRecord] = []
        source_versions = [version for version in self.state.versions.values() if version.source_id == source_id]
        for version in source_versions:
            if version.parsed_path is None:
                version.document_id = document_id
                version.identity_status = "resolved"
                version.identity_resolution_method = method
                version.identity_confidence = confidence
                continue
            migration_id = content_hash_for_bytes(f"{source_id}\n{version.version_id}\n{document_id}".encode())
            existing = self.state.migrations.get(migration_id)
            if existing is None:
                source_path = version.parsed_path
                storage_document_id = self._storage_segment(document_id)
                destination_path = f"parsed/documents/{storage_document_id}/{version.version_id}.md"
                expected_hash = content_hash_for_bytes(self.resolve_path(source_path).read_bytes())
                existing = MigrationRecord(
                    migration_id=migration_id,
                    source_id=source_id,
                    version_id=version.version_id,
                    document_id=document_id,
                    source_path=source_path,
                    destination_path=destination_path,
                    expected_content_hash=expected_hash,
                    resolution_evidence=evidence,
                    identity_resolution_method=method,
                    identity_confidence=confidence,
                )
                self.state.migrations[migration_id] = existing
                self.save()
            self._advance_migration(existing)
            migrations.append(existing)

        resolved = all(
            version.document_id == document_id and version.identity_status == "resolved" for version in source_versions
        )
        if resolved:
            source.document_id = document_id
            source.identity_status = "resolved"
            source.identity_resolution_method = method
            source.identity_confidence = confidence
        self.save()
        return migrations

    def reconcile_migrations(self) -> list[MigrationRecord]:
        reconciled: list[MigrationRecord] = []
        for migration in self.state.migrations.values():
            if migration.state in {"pending", "file_moved"}:
                self._advance_migration(migration)
                reconciled.append(migration)
            elif migration.state == "committed":
                destination = self.resolve_path(migration.destination_path)
                if (
                    not destination.is_file()
                    or content_hash_for_bytes(destination.read_bytes()) != migration.expected_content_hash
                ):
                    self._quarantine(migration, "committed destination is missing or has changed")
                    reconciled.append(migration)
            self._finalize_source_resolution(migration)
        return reconciled

    def _finalize_source_resolution(self, migration: MigrationRecord) -> None:
        source = self.state.sources[migration.source_id]
        source_versions = [version for version in self.state.versions.values() if version.source_id == source.source_id]
        resolved = all(
            version.document_id == migration.document_id and version.identity_status == "resolved"
            for version in source_versions
        )
        if resolved:
            source.document_id = migration.document_id
            source.identity_status = "resolved"
            source.identity_resolution_method = migration.identity_resolution_method
            source.identity_confidence = migration.identity_confidence
            self.save()

    def _advance_migration(self, migration: MigrationRecord) -> None:
        migration.attempts += 1
        source_path = self.resolve_path(migration.source_path)
        destination_path = self.resolve_path(migration.destination_path)
        try:
            if migration.state == "pending":
                self._move_and_verify(source_path, destination_path, migration.expected_content_hash)
                migration.state = "file_moved"
                migration.former_path = migration.source_path
                migration.last_error = None
                self.save()
            if migration.state == "file_moved":
                if not destination_path.is_file():
                    raise FileNotFoundError(destination_path)
                destination_hash = content_hash_for_bytes(destination_path.read_bytes())
                if destination_hash != migration.expected_content_hash:
                    self._quarantine(migration, "destination content hash mismatch")
                    return
                version = self.state.versions[migration.version_id]
                version.document_id = migration.document_id
                version.identity_status = "resolved"
                version.identity_resolution_method = migration.identity_resolution_method
                version.identity_confidence = migration.identity_confidence
                version.parsed_path = migration.destination_path
                migration.state = "committed"
                migration.resolution_timestamp = utc_now()
                migration.last_error = None
                self.save()
        except (OSError, ValueError) as exc:
            migration.last_error = str(exc)
            destination_mismatch = (
                destination_path.exists()
                and content_hash_for_bytes(destination_path.read_bytes()) != migration.expected_content_hash
            )
            if destination_mismatch:
                self._quarantine(migration, str(exc))
            else:
                self.save()

    def _move_and_verify(self, source: Path, destination: Path, expected_hash: str) -> None:
        if destination.exists():
            if content_hash_for_bytes(destination.read_bytes()) != expected_hash:
                raise ValueError("destination content hash mismatch")
            if source.exists() and source != destination:
                source.unlink()
            return
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.migrating")
        shutil.copy2(source, temporary)
        if content_hash_for_bytes(temporary.read_bytes()) != expected_hash:
            temporary.unlink(missing_ok=True)
            raise ValueError("copied artifact content hash mismatch")
        os.replace(temporary, destination)
        if content_hash_for_bytes(destination.read_bytes()) != expected_hash:
            raise ValueError("destination content hash mismatch")
        source.unlink()

    def _quarantine(self, migration: MigrationRecord, error: str) -> None:
        migration.state = "quarantined"
        migration.last_error = error
        self.save()

    def resolve_path(self, relative_path: str) -> Path:
        candidate = (self.site_root / relative_path).resolve()
        if not candidate.is_relative_to(self.site_root):
            raise ValueError(f"Manifest path escapes site root: {relative_path}")
        return candidate

    @staticmethod
    def _storage_segment(identifier: str) -> str:
        if identifier not in {".", ".."} and re.fullmatch(r"[A-Za-z0-9._-]+", identifier):
            return identifier
        return content_hash_for_bytes(identifier.encode("utf-8"))

    def version_for_local_path(self, path: Path) -> ManifestRecord | None:
        try:
            relative = path.resolve().relative_to(self.site_root).as_posix()
        except ValueError:
            return None
        return next((version for version in self.state.versions.values() if version.local_path == relative), None)

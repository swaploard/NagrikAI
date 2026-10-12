from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

NORMALIZED_DOCUMENT_SCHEMA_VERSION: Literal["1.0"] = "1.0"

SourceFormat = Literal["html", "pdf", "json", "rendered", "unsupported"]
FetchStatus = Literal["success", "unchanged", "reused", "blocked", "failed", "incomplete", "unavailable"]
QualityStatus = Literal[
    "complete",
    "metadata_only",
    "low_text",
    "ocr_required",
    "unsupported",
    "incomplete",
    "failed",
]
ExtractionMethod = Literal["html", "pymupdf", "json", "rendered", "viewer_link", "ocr", "none"]
SourceVersionStatus = Literal["current_source", "historical_source"]
LegalApplicabilityStatus = Literal["applicable", "not_applicable", "conditional", "expired", "unknown"]


class StrictSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(StrictSchemaModel):
    source_url: str
    method: str
    captured_at: datetime
    field: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class LegalFact(StrictSchemaModel):
    value: str | None = None
    status: Literal["explicit", "unresolved", "not_present"] = "unresolved"
    provenance: list[Provenance] = Field(default_factory=list)


class LegalApplicability(StrictSchemaModel):
    status: LegalApplicabilityStatus = "unknown"
    effective_date: LegalFact = Field(default_factory=LegalFact)
    rescission_or_supersession_date: LegalFact = Field(default_factory=LegalFact)
    applicable_goods_services: list[LegalFact] = Field(default_factory=list)
    conditions: list[LegalFact] = Field(default_factory=list)
    jurisdiction: LegalFact = Field(default_factory=LegalFact)


class LegalMetadata(StrictSchemaModel):
    document_type: str | None = None
    document_number: str | None = None
    document_year: int | None = None
    publication_date: LegalFact = Field(default_factory=LegalFact)
    tax_type: str | None = None
    notification_subtype: str | None = None
    subject: str | None = None
    language: str = "en"
    amendment_document_ids: list[str] = Field(default_factory=list)
    referenced_document_urls: list[str] = Field(default_factory=list)
    hsn_sac_codes: list[str] = Field(default_factory=list)
    applicability: LegalApplicability = Field(default_factory=LegalApplicability)


class TableRow(StrictSchemaModel):
    row_index: int = Field(ge=0)
    cells: dict[str, str]
    heading: str | None = None
    schedule: str | None = None
    condition: str | None = None
    footnotes: list[str] = Field(default_factory=list)
    page_number: int | None = Field(default=None, ge=1)


class TablePayload(StrictSchemaModel):
    table_id: str
    headers: list[str]
    rows: list[TableRow]
    title: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    footnotes: list[str] = Field(default_factory=list)


class PagePayload(StrictSchemaModel):
    page_number: int = Field(ge=1)
    text: str
    table_ids: list[str] = Field(default_factory=list)


class ExtractionInfo(StrictSchemaModel):
    method: ExtractionMethod
    parser_name: str
    parser_version: str
    quality_status: QualityStatus
    ocr_used: bool = False
    errors: list[str] = Field(default_factory=list)


class NormalizedDocument(StrictSchemaModel):
    """Versioned representation shared by all fetched source formats."""

    schema_version: Literal["1.0"] = NORMALIZED_DOCUMENT_SCHEMA_VERSION
    source_id: str
    document_id: str
    version_id: str
    observed_url: str
    canonical_url: str
    site: str
    language: str
    media_type: str
    source_format: SourceFormat
    raw_local_path: str
    content_hash: str
    fetch_status: FetchStatus
    fetched_at: datetime
    title: str = ""
    source_version_status: SourceVersionStatus = "current_source"
    extraction: ExtractionInfo
    pages: list[PagePayload] = Field(default_factory=list)
    tables: list[TablePayload] = Field(default_factory=list)
    legal_metadata: LegalMetadata = Field(default_factory=LegalMetadata)
    provenance: list[Provenance] = Field(default_factory=list)
    discovered_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    official_file_url: str | None = None
    errors: list[str] = Field(default_factory=list)

    @field_validator("source_id", "document_id", "version_id", "content_hash")
    @classmethod
    def require_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identity and hash fields cannot be blank")
        return value


class DiscoveryCandidate(StrictSchemaModel):
    """Strict metadata-only input accepted from discovery spiders."""

    schema_version: Literal["1.0"] = NORMALIZED_DOCUMENT_SCHEMA_VERSION
    kind: Literal["discovery_candidate"] = "discovery_candidate"
    source_id: str
    document_id: str
    record_id: str
    portal_record_id: str
    content_id: str = ""
    observed_url: str
    canonical_url: str
    listing_url: str
    site: str
    category: str
    tax_type: str
    language: str
    title: str
    download_url: str | None = None
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    raw_record: dict[str, Any]


class Document:
    def __init__(
        self,
        *,
        content: str,
        source: str,
        site: str,
        title: str = "",
        url: str = "",
        crawled_at: datetime | None = None,
        doc_id: str | None = None,
    ) -> None:
        self.doc_id = doc_id or uuid4().hex
        self.content = content
        self.source = source
        self.site = site
        self.title = title
        self.url = url
        self.crawled_at = crawled_at or datetime.now()

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "source": self.source,
            "site": self.site,
            "title": self.title,
            "url": self.url,
            "crawled_at": self.crawled_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        return cls(
            content=data["content"],
            source=data["source"],
            site=data["site"],
            title=data.get("title", ""),
            url=data.get("url", ""),
            crawled_at=datetime.fromisoformat(data["crawled_at"]) if data.get("crawled_at") else None,
            doc_id=data.get("doc_id"),
        )

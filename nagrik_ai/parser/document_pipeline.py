from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin

from lxml import html

from nagrik_ai.crawler.crawl_manifest import CrawlManifest, ManifestRecord
from nagrik_ai.crawler.url_utils import canonicalize_url, document_id_for_identifier
from nagrik_ai.models.document import (
    ExtractionInfo,
    LegalMetadata,
    NormalizedDocument,
    PagePayload,
    Provenance,
    SourceFormat,
)
from nagrik_ai.parser.pdf_parser import PDFParser

NORMALIZER_NAME = "NagrikDocumentPipeline"
NORMALIZER_VERSION = "1.0"


def classify_source(content_type: str, content: bytes) -> tuple[SourceFormat, str]:
    declared = content_type.partition(";")[0].strip().casefold()
    prefix = content[:1024].lstrip()
    if content.startswith(b"%PDF-"):
        return "pdf", "application/pdf"
    if declared in {"application/json", "application/ld+json"} or declared.endswith("+json"):
        try:
            json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return "unsupported", declared or "application/octet-stream"
        return "json", declared
    if declared in {"text/html", "application/xhtml+xml"} or re.search(
        br"<(?:!doctype\s+html|html|body)(?:\s|>)", prefix, re.IGNORECASE
    ):
        return "html", declared or "text/html"
    if declared == "application/pdf":
        return "unsupported", declared
    return "unsupported", declared or "application/octet-stream"


def normalize_source(
    manifest: CrawlManifest,
    version: ManifestRecord,
    content: bytes,
    *,
    document_id: str | None = None,
    language: str = "en",
    discovered_metadata: dict[str, str | int | float | bool | None] | None = None,
    official_file_url: str | None = None,
    provenance_url: str | None = None,
    provenance_method: str = "crawl_response",
    pdf_parser: PDFParser | None = None,
) -> tuple[NormalizedDocument, Path]:
    source_format, media_type = classify_source(version.content_type, content)
    resolved_document_id = document_id or version.document_id or document_id_for_identifier(
        version.site, f"source:{version.source_id}"
    )
    provenance = [
        Provenance(
            source_url=provenance_url or version.observed_source_url,
            method=provenance_method,
            captured_at=datetime.fromisoformat(version.fetch_time),
        )
    ]
    pages: list[PagePayload] = []
    tables = []
    extracted_file_url: str | None = official_file_url
    errors: list[str] = []
    extraction = ExtractionInfo(
        method="none",
        parser_name=NORMALIZER_NAME,
        parser_version=NORMALIZER_VERSION,
        quality_status="unsupported",
    )

    if source_format == "pdf":
        parser = pdf_parser or PDFParser()
        pdf_result = parser.extract(content)
        pages = pdf_result.pages
        tables = pdf_result.tables
        errors.extend(pdf_result.errors)
        extraction = ExtractionInfo(
            method="ocr" if pdf_result.ocr_used else "pymupdf",
            parser_name=parser.parser_name,
            parser_version=parser.parser_version,
            quality_status=pdf_result.quality_status,
            ocr_used=pdf_result.ocr_used,
            errors=pdf_result.errors,
        )
    elif source_format == "html":
        decoded = content.decode("utf-8", errors="replace")
        tree: Any = html.fromstring(decoded)
        text = "\n".join(part.strip() for part in tree.text_content().splitlines() if part.strip())
        links = cast(list[Any], tree.xpath("//a/@href"))
        extracted_file_url = official_file_url or next(
            (str(link) for link in links if str(link).casefold().endswith(".pdf")), None
        )
        if extracted_file_url:
            extracted_file_url = urljoin(version.observed_source_url, extracted_file_url)
        pages = [PagePayload(page_number=1, text=text)]
        extraction = ExtractionInfo(
            method="viewer_link" if extracted_file_url else "html",
            parser_name=NORMALIZER_NAME,
            parser_version=NORMALIZER_VERSION,
            quality_status="complete" if text else "incomplete",
        )
    elif source_format == "json":
        payload = json.loads(content)
        pages = [PagePayload(page_number=1, text=json.dumps(payload, ensure_ascii=False, sort_keys=True))]
        extraction = ExtractionInfo(
            method="json",
            parser_name=NORMALIZER_NAME,
            parser_version=NORMALIZER_VERSION,
            quality_status="metadata_only",
        )
    else:
        errors.append(f"Unsupported content type/signature: {version.content_type}")

    normalized = NormalizedDocument(
        source_id=version.source_id,
        document_id=resolved_document_id,
        version_id=version.version_id,
        observed_url=version.observed_source_url,
        canonical_url=canonicalize_url(version.canonical_source_url),
        site=version.site,
        language=language,
        media_type=media_type,
        source_format=source_format,
        raw_local_path=version.local_path,
        content_hash=version.content_hash,
        fetch_status=version.status,
        fetched_at=datetime.fromisoformat(version.fetch_time),
        title=version.title,
        extraction=extraction,
        pages=pages,
        tables=tables,
        legal_metadata=LegalMetadata(language=language),
        provenance=provenance,
        discovered_metadata=discovered_metadata or {},
        official_file_url=extracted_file_url,
        errors=errors,
    )
    storage_document_id = manifest._storage_segment(resolved_document_id)
    output = manifest.site_root / "parsed" / "records" / storage_document_id / f"{version.version_id}.json"
    CrawlManifest._atomic_write(
        output,
        (json.dumps(normalized.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(),
    )
    return normalized, output


def mark_previous_normalized_version_historical(manifest: CrawlManifest, version: ManifestRecord) -> Path | None:
    previous_id = version.previous_version_id
    if previous_id is None:
        return None
    matches = list((manifest.site_root / "parsed" / "records").glob(f"*/{previous_id}.json"))
    if not matches:
        return None
    path = matches[0]
    previous = NormalizedDocument.model_validate_json(path.read_text(encoding="utf-8"))
    historical = previous.model_copy(update={"source_version_status": "historical_source"})
    CrawlManifest._atomic_write(
        path,
        (json.dumps(historical.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(),
    )
    return path

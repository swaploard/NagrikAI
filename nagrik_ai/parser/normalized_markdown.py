from __future__ import annotations

import json

import yaml

from nagrik_ai.models.document import NormalizedDocument, TablePayload
from nagrik_ai.parser.gst_document_parser import GSTDocumentParser


def render_normalized_markdown(document: NormalizedDocument) -> str:
    metadata = {
        "schema_version": document.schema_version,
        "document_id": document.document_id,
        "source_id": document.source_id,
        "version_id": document.version_id,
        "content_hash": document.content_hash,
        "source_format": document.source_format,
        "parser_name": GSTDocumentParser.parser_name,
        "parser_version": GSTDocumentParser.parser_version,
        "extraction_method": document.extraction.method,
        "extraction_parser_name": document.extraction.parser_name,
        "extraction_parser_version": document.extraction.parser_version,
        "source_url": document.canonical_url,
        "language": document.language,
        "source_version_status": document.source_version_status,
        "document_type": document.legal_metadata.document_type,
        "document_number": document.legal_metadata.document_number,
        "publication_date": document.legal_metadata.publication_date.value,
        "legal_applicability_status": document.legal_metadata.applicability.status,
        "applicability_start": document.legal_metadata.applicability.effective_date.value,
        "applicability_end": document.legal_metadata.applicability.rescission_or_supersession_date.value,
        "legal_metadata": document.legal_metadata.model_dump(mode="json"),
    }
    body: list[str] = [f"# {document.title or document.legal_metadata.subject or 'Untitled'}"]
    for page in document.pages:
        body.extend(("", f"<!-- page:{page.page_number} -->", f"## Page {page.page_number}", "", page.text))
        for table_id in page.table_ids:
            table = next((candidate for candidate in document.tables if candidate.table_id == table_id), None)
            if table is not None:
                body.extend(("", _render_table(table)))
    unpaged = [table for table in document.tables if table.page_number is None]
    for table in unpaged:
        body.extend(("", _render_table(table)))
    front_matter = yaml.safe_dump(metadata, sort_keys=True, allow_unicode=True)
    return f"---\n{front_matter}---\n\n" + "\n".join(body).strip() + "\n"


def _render_table(table: TablePayload) -> str:
    payload = json.dumps(table.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return f"<!-- structured-table:{table.table_id}\n{payload}\n-->"

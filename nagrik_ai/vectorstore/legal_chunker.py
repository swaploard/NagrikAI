from __future__ import annotations

import hashlib
import json
import re
from typing import ClassVar, Literal

from langchain_core.documents import Document

from nagrik_ai.models.document import NormalizedDocument

RetrievalScope = Literal["current_source", "historical_source", "legally_applicable", "all_versions"]


class LegalChunker:
    name = "LegalChunker"
    version = "1.0"
    _boundaries: ClassVar[dict[str, re.Pattern[str]]] = {
        "act": re.compile(r"(?im)^(?=\s*(?:section\s+)?\d+[A-Z]?\s*[.:-])"),
        "rule": re.compile(r"(?im)^(?=\s*rule\s+\d+[A-Z]?\b)"),
        "notification": re.compile(r"(?m)^(?=\s*\(?\d+[A-Za-z]?\)?[.)]\s+)"),
        "circular": re.compile(r"(?m)^(?=\s*\d+(?:\.\d+)*[.)]?\s+)"),
        "order": re.compile(r"(?m)^(?=\s*\(?\d+[A-Za-z]?\)?[.)]\s+)"),
        "instruction": re.compile(r"(?m)^(?=\s*\d+(?:\.\d+)*[.)]?\s+)"),
    }

    def chunk_document(self, document: NormalizedDocument) -> list[Document]:
        chunks: list[Document] = []
        source_type = document.legal_metadata.document_type or "unstructured"
        for page in document.pages:
            parts = self._split(page.text, source_type)
            for part_index, text in enumerate(parts):
                location = f"page:{page.page_number}:part:{part_index}"
                chunks.append(self._chunk(document, text, location, page.page_number, source_type))
        for table in document.tables:
            for row in table.rows:
                inherited = " | ".join(value for value in (row.schedule, row.heading, row.condition) if value)
                row_text = " | ".join(f"{key}: {value}" for key, value in row.cells.items())
                text = f"{inherited}\n{row_text}".strip()
                location = f"table:{table.table_id}:row:{row.row_index}"
                chunks.append(self._chunk(document, text, location, row.page_number, "schedule"))
        return chunks

    def chunk_markdown(self, content: str, metadata: dict[str, object]) -> list[Document]:
        document_id = str(metadata["document_id"])
        version_id = str(metadata["version_id"])
        source_type = str(metadata.get("document_type") or metadata.get("source_type") or "unstructured")
        source_status = str(metadata.get("source_version_status") or "current_source")
        applicability_status = str(metadata.get("legal_applicability_status") or "unknown")
        scopes = {"all_versions", source_status}
        if applicability_status == "applicable":
            scopes.add("legally_applicable")
        table_payloads = re.findall(r"<!-- structured-table:[^\n]+\n(.*?)\n-->", content, re.DOTALL)
        prose = re.sub(r"<!-- structured-table:[^\n]+\n.*?\n-->", "", content, flags=re.DOTALL)
        pages = re.split(r"(?=<!-- page:\d+ -->)", prose)
        chunks: list[Document] = []
        for page_index, page in enumerate(pages):
            if not page.strip():
                continue
            page_match = re.search(r"<!-- page:(\d+) -->", page)
            page_number = int(page_match.group(1)) if page_match else page_index + 1
            for part_index, part in enumerate(self._split(page, source_type)):
                location = f"page:{page_number}:part:{part_index}"
                chunk_id = self.chunk_id(document_id, version_id, location)
                chunk_metadata = dict(metadata)
                chunk_metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "logical_location": location,
                        "page": page_number,
                        "chunker_name": self.name,
                        "chunker_version": self.version,
                        "source_version_status": source_status,
                        "legal_applicability_status": applicability_status,
                        "retrieval_scopes": sorted(scopes),
                        "citation_url": metadata.get("source_url", ""),
                    }
                )
                chunks.append(Document(page_content=part, metadata=chunk_metadata))
        for raw_table in table_payloads:
            payload = json.loads(raw_table)
            table_id = str(payload["table_id"])
            for row in payload.get("rows", []):
                location = f"table:{table_id}:row:{row['row_index']}"
                inherited = " | ".join(
                    str(row.get(key, "")) for key in ("schedule", "heading", "condition") if row.get(key)
                )
                cells = " | ".join(f"{key}: {value}" for key, value in row.get("cells", {}).items())
                chunk_metadata = dict(metadata)
                chunk_metadata.update(
                    {
                        "chunk_id": self.chunk_id(document_id, version_id, location),
                        "logical_location": location,
                        "page": int(row.get("page_number") or payload.get("page_number") or 0),
                        "chunker_name": self.name,
                        "chunker_version": self.version,
                        "source_version_status": source_status,
                        "legal_applicability_status": applicability_status,
                        "retrieval_scopes": sorted(scopes),
                        "citation_url": metadata.get("source_url", ""),
                        "source_type": "schedule",
                    }
                )
                chunks.append(Document(page_content=f"{inherited}\n{cells}".strip(), metadata=chunk_metadata))
        return chunks

    def _chunk(
        self,
        document: NormalizedDocument,
        text: str,
        location: str,
        page_number: int | None,
        source_type: str,
    ) -> Document:
        applicability = document.legal_metadata.applicability
        scopes: list[RetrievalScope] = ["all_versions", document.source_version_status]
        if applicability.status == "applicable":
            scopes.append("legally_applicable")
        metadata: dict[str, object] = {
            "chunk_id": self.chunk_id(document.document_id, document.version_id, location),
            "document_id": document.document_id,
            "source_id": document.source_id,
            "version_id": document.version_id,
            "content_hash": document.content_hash,
            "document_number": document.legal_metadata.document_number or "",
            "publication_date": document.legal_metadata.publication_date.value or "",
            "page": page_number or 0,
            "citation_url": document.canonical_url,
            "source_url": document.canonical_url,
            "source_type": source_type,
            "logical_location": location,
            "schema_version": document.schema_version,
            "parser_name": document.extraction.parser_name,
            "parser_version": document.extraction.parser_version,
            "chunker_name": self.name,
            "chunker_version": self.version,
            "source_version_status": document.source_version_status,
            "legal_applicability_status": applicability.status,
            "applicability_start": applicability.effective_date.value or "",
            "applicability_end": applicability.rescission_or_supersession_date.value or "",
            "retrieval_scopes": sorted(set(scopes)),
        }
        return Document(page_content=text, metadata=metadata)

    @classmethod
    def _split(cls, text: str, source_type: str) -> list[str]:
        pattern = cls._boundaries.get(source_type.casefold())
        if pattern is not None:
            parts = [part.strip() for part in pattern.split(text) if part.strip()]
        else:
            parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        return parts or ([text.strip()] if text.strip() else [])

    @staticmethod
    def chunk_id(document_id: str, version_id: str, logical_location: str) -> str:
        return hashlib.sha256(f"{document_id}\n{version_id}\n{logical_location}".encode()).hexdigest()

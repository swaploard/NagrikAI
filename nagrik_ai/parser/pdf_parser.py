from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import pymupdf

from nagrik_ai.models.document import PagePayload, QualityStatus, TablePayload, TableRow

OCRCallback = Callable[[bytes, int], str]


@dataclass(frozen=True)
class PDFExtractionResult:
    pages: list[PagePayload] = field(default_factory=list)
    tables: list[TablePayload] = field(default_factory=list)
    quality_status: QualityStatus = "complete"
    extraction_method: str = "pymupdf"
    ocr_used: bool = False
    errors: list[str] = field(default_factory=list)


class PDFParser:
    parser_name = "PyMuPDF"
    parser_version = "1.1"

    def __init__(self, usefulness_threshold: int = 80, ocr: OCRCallback | None = None) -> None:
        self.usefulness_threshold = usefulness_threshold
        self.ocr = ocr

    def extract(self, pdf_bytes: bytes) -> PDFExtractionResult:
        pages: list[PagePayload] = []
        tables: list[TablePayload] = []
        errors: list[str] = []
        try:
            document = pymupdf.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[no-untyped-call]
        except Exception as exc:
            return PDFExtractionResult(quality_status="failed", errors=[str(exc)])

        try:
            table_context: dict[str, Any] = {}
            for page_index, page in enumerate(cast(Any, document)):
                page_number = page_index + 1
                text = str(page.get_text("text", sort=True)).strip()
                page_tables = self._extract_tables(page, page_number, errors, table_context)
                tables.extend(page_tables)
                pages.append(
                    PagePayload(
                        page_number=page_number,
                        text=text,
                        table_ids=[table.table_id for table in page_tables],
                    )
                )
        finally:
            document.close()  # type: ignore[no-untyped-call]

        total_text = sum(len(page.text.strip()) for page in pages)
        if total_text >= self.usefulness_threshold:
            return PDFExtractionResult(pages=pages, tables=tables, errors=errors)
        if self.ocr is None:
            return PDFExtractionResult(
                pages=pages,
                tables=tables,
                quality_status="ocr_required",
                errors=errors,
            )

        ocr_pages: list[PagePayload] = []
        for page in pages:
            try:
                ocr_text = self.ocr(pdf_bytes, page.page_number).strip()
            except Exception as exc:
                errors.append(f"OCR page {page.page_number}: {exc}")
                ocr_text = ""
            ocr_pages.append(page.model_copy(update={"text": ocr_text or page.text}))
        ocr_text_length = sum(len(page.text.strip()) for page in ocr_pages)
        quality: QualityStatus = "complete" if ocr_text_length >= self.usefulness_threshold else "low_text"
        return PDFExtractionResult(
            pages=ocr_pages,
            tables=tables,
            quality_status=quality,
            extraction_method="ocr",
            ocr_used=True,
            errors=errors,
        )

    @staticmethod
    def _extract_tables(
        page: Any, page_number: int, errors: list[str], context: dict[str, Any] | None = None
    ) -> list[TablePayload]:
        context = context if context is not None else {}
        try:
            finder = page.find_tables()
        except Exception as exc:
            errors.append(f"Table extraction page {page_number}: {exc}")
            return []

        result: list[TablePayload] = []
        for table_index, table in enumerate(getattr(finder, "tables", [])):
            matrix = table.extract()
            if not matrix:
                continue
            first_row = [str(value or "").strip() for value in matrix[0]]
            is_header = not re.fullmatch(r"\(?\d+\)?\.?", first_row[0]) and any(
                re.search(r"description|chapter|heading|s\.?\s*no|hsn|sac|rate", value, re.I)
                for value in first_row
            )
            if is_header:
                headers = [value or f"column_{index + 1}" for index, value in enumerate(first_row)]
                context["headers"] = headers
                data_rows = matrix[1:]
            else:
                prior_headers = context.get("headers", [])
                headers = prior_headers if len(prior_headers) == len(first_row) else [
                    f"column_{index + 1}" for index in range(len(first_row))
                ]
                data_rows = matrix
            preceding = str(page.get_text("text", clip=(0, 0, page.rect.width, table.bbox[1]), sort=True))
            if re.search(r"Central\s+Tax", preceding, re.I):
                context["rate_field"] = "Central tax rate"
            schedules = re.findall(r"Schedule\s+[IVX]+\s*[-\u2013\u2014]\s*\d+(?:\.\d+)?\s*%", preceding, re.I)
            if schedules:
                context["schedule"] = schedules[-1].strip()
            schedule = context.get("schedule")
            rate_match = re.search(r"(\d+(?:\.\d+)?\s*%)", schedule or "")
            rows: list[TableRow] = []
            for row_index, values in enumerate(data_rows):
                if all(re.fullmatch(r"\(?\d+\)?", str(value or "").strip()) for value in values):
                    continue
                cells = {
                    header: str(values[index] or "").strip() if index < len(values) else ""
                    for index, header in enumerate(headers)
                }
                if rate_match:
                    cells[str(context.get("rate_field", "Schedule rate"))] = rate_match.group(1)
                rows.append(TableRow(
                    row_index=row_index, cells=cells, page_number=page_number,
                    schedule=schedule, heading=schedule,
                ))
            identity = f"{page_number}:{table_index}:{headers}:{matrix}".encode()
            table_id = hashlib.sha256(identity).hexdigest()
            result.append(
                TablePayload(
                    table_id=table_id,
                    headers=headers,
                    rows=rows,
                    page_number=page_number,
                )
            )
        return result

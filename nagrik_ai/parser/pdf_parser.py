from __future__ import annotations

import hashlib
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
    parser_version = "1.0"

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
            for page_index, page in enumerate(cast(Any, document)):
                page_number = page_index + 1
                text = str(page.get_text("text", sort=True)).strip()
                page_tables = self._extract_tables(page, page_number, errors)
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
    def _extract_tables(page: Any, page_number: int, errors: list[str]) -> list[TablePayload]:
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
            headers = [str(value or "").strip() or f"column_{index + 1}" for index, value in enumerate(matrix[0])]
            rows: list[TableRow] = []
            for row_index, values in enumerate(matrix[1:]):
                cells = {
                    header: str(values[index] or "").strip() if index < len(values) else ""
                    for index, header in enumerate(headers)
                }
                rows.append(TableRow(row_index=row_index, cells=cells, page_number=page_number))
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

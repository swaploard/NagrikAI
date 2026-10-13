from __future__ import annotations

import re
from datetime import datetime
from typing import ClassVar

from nagrik_ai.crawler.url_utils import document_id_for_identifier
from nagrik_ai.models.document import LegalApplicability, LegalFact, LegalMetadata, NormalizedDocument, Provenance


class GSTDocumentParser:
    parser_name = "GSTDocumentParser"
    parser_version = "1.1"

    _number_patterns: ClassVar[dict[str, re.Pattern[str]]] = {
        "notification": re.compile(
            r"\b(?:Notification\s+No\.?\s*)?([0-9A-Za-z.-]+/[0-9]{4}(?:-[A-Za-z][A-Za-z ()-]*)?)",
            re.I,
        ),
        "circular": re.compile(
            r"\b(?:Circular\s+No\.?\s*)?([0-9A-Za-z.-]+/[0-9A-Za-z.-]+/[0-9]{4}(?:-[A-Za-z-]+)?)",
            re.I,
        ),
        "instruction": re.compile(r"\b(Instruction\s+No\.?\s*[0-9A-Za-z./-]+)", re.I),
        "order": re.compile(r"\b(Order\s+No\.?\s*[0-9A-Za-z./-]+)", re.I),
    }
    _date_pattern = (
        r"([0-3]?\d[-/.][01]?\d[-/.](?:19|20)\d{2}|"
        r"[0-3]?\d(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+(?:19|20)\d{2})"
    )

    def parse(self, document: NormalizedDocument) -> NormalizedDocument:
        text = "\n".join(page.text for page in document.pages)
        category = str(document.discovered_metadata.get("category", "")).casefold()
        document_type = self._document_type(text, category)
        document_number = self._metadata_number(document) or self._document_number(text, document_type)
        publication = self._explicit_fact(
            text,
            rf"(?:dated|date of issue|notification date|publication date)\s*[:\-]?\s*{self._date_pattern}",
            document,
            "publication_date",
        )
        portal_date = document.discovered_metadata.get("date")
        if isinstance(portal_date, str):
            try:
                publication_date = datetime.fromisoformat(portal_date).date().isoformat()
            except ValueError:
                pass
            else:
                publication = LegalFact(
                    value=publication_date,
                    status="explicit",
                    provenance=[Provenance(
                        source_url=str(document.discovered_metadata.get("listing_url") or document.canonical_url),
                        method="portal_metadata", captured_at=document.fetched_at,
                        field="publication_date", confidence=1.0,
                    )],
                )
        effective = self._explicit_fact(
            text,
            rf"(?:come|comes|shall come) into force (?:with effect from|from|on)?\s*{self._date_pattern}",
            document,
            "effective_date",
        )
        rescission = self._explicit_fact(
            text,
            rf"(?:rescinded|superseded)(?:\s+with effect from|\s+on)?\s*{self._date_pattern}",
            document,
            "rescission_or_supersession_date",
        )
        references = sorted(set(re.findall(r"https?://[^\s)>\]]+", text)))
        amendment_ids = [
            document_id_for_identifier("cbic-reference", url)
            for url in references
            if "amend" in url.casefold() or "view-pdf" in url.casefold()
        ]
        codes = sorted(
            set(
                match.group(1)
                for match in re.finditer(r"\b(?:HSN|SAC)(?:\s+Code)?\s*[:\-]?\s*([0-9]{2,8})\b", text, re.I)
            )
        )
        subtype = self._optional_string(document.discovered_metadata.get("subtype"))
        subject = self._optional_string(document.discovered_metadata.get("subject")) or document.title or None
        legal = LegalMetadata(
            document_type=document_type,
            document_number=document_number,
            document_year=self._year(document_number),
            publication_date=publication,
            tax_type=self._optional_string(document.discovered_metadata.get("tax_type")) or "GST",
            notification_subtype=subtype,
            subject=subject,
            language=document.language,
            amendment_document_ids=amendment_ids,
            referenced_document_urls=references,
            hsn_sac_codes=codes,
            applicability=LegalApplicability(
                status="unknown" if effective.status != "explicit" else "applicable",
                effective_date=effective,
                rescission_or_supersession_date=rescission,
            ),
        )
        return document.model_copy(update={"legal_metadata": legal})

    def _explicit_fact(
        self,
        text: str,
        pattern: str,
        document: NormalizedDocument,
        field: str,
    ) -> LegalFact:
        match = re.search(pattern, text, re.I)
        if match is None:
            return LegalFact(status="unresolved")
        value = match.group(match.lastindex or 1)
        provenance = Provenance(
            source_url=document.canonical_url,
            method="explicit_text",
            captured_at=document.fetched_at,
            field=field,
            confidence=1.0,
        )
        return LegalFact(value=value.strip(), status="explicit", provenance=[provenance])

    def _document_number(self, text: str, document_type: str | None) -> str | None:
        if document_type is None:
            return None
        pattern = self._number_patterns.get(document_type)
        if pattern is None:
            return None
        match = pattern.search(text)
        if match is None:
            return None
        value = re.split(r"\s+(?:dated|date of issue|publication date)\b", match.group(1), maxsplit=1, flags=re.I)[0]
        return value.strip(" ,.;")

    @staticmethod
    def _document_type(text: str, category: str) -> str | None:
        combined = f"{category}\n{text[:1000]}".casefold()
        for value in ("notification", "circular", "instruction", "order", "rule", "act"):
            if value in combined:
                return value
        return None

    @staticmethod
    def _metadata_number(document: NormalizedDocument) -> str | None:
        return GSTDocumentParser._optional_string(document.discovered_metadata.get("number"))

    @staticmethod
    def _year(document_number: str | None) -> int | None:
        if document_number is None:
            return None
        match = re.search(r"\b((?:19|20)\d{2})\b", document_number)
        return int(match.group(1)) if match else None

    @staticmethod
    def _optional_string(value: object) -> str | None:
        return str(value).strip() if value not in (None, "") else None

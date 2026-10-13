from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, cast
from urllib.parse import urlencode, urljoin

import scrapy
from scrapy.http import Response

from nagrik_ai.config.config_models import SiteConfig
from nagrik_ai.crawler.spiders.site_spider import SiteSpider
from nagrik_ai.crawler.url_utils import document_id_for_identifier, source_id_for_url

logger = logging.getLogger(__name__)


class CbicTaxInfoSpider(SiteSpider):
    """Discover GST records through the portal's verified public JSON services."""

    name = "cbic_taxinfo"
    _paged_categories: ClassVar[set[str]] = {"notifications", "circulars", "instructions", "orders"}
    _tab_names: ClassVar[dict[str, str]] = {
        "notifications": "Notifications",
        "circulars": "Circulars",
        "instructions": "Instructions",
        "orders": "Orders",
    }
    _title_fields: ClassVar[dict[str, tuple[str, ...]]] = {
        "notifications": ("notificationName", "notificationNo"),
        "circulars": ("circularName", "circularNo"),
        "instructions": ("instructionName", "instructionNo"),
        "orders": ("orderName", "orderNo"),
        "acts": ("actName",),
        "rules": ("ruleDocName",),
    }
    _number_fields: ClassVar[dict[str, str]] = {
        "notifications": "notificationNo",
        "circulars": "circularNo",
        "instructions": "instructionNo",
        "orders": "orderNo",
    }

    def __init__(
        self,
        site_config: SiteConfig,
        output_dir: Path,
        manage: bool = False,
        force: bool = False,
        incremental: bool = False,
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__(site_config, output_dir, manage, force, incremental, *args, **kwargs)
        self._seen_record_keys: set[str] = set()
        self._candidate_count = 0

    def start_requests(self) -> Iterator[scrapy.Request]:
        discovery = self.site_config.discovery
        for category in discovery.document_categories:
            route = discovery.api_routes.get(category)
            if route is None:
                logger.warning("No verified API route configured for %s", category)
                continue
            if category in self._paged_categories:
                filters: list[str | None] = list(discovery.category_filters.get(category, [])) or [None]
                for year in discovery.years:
                    for subtype in filters:
                        yield self._listing_request(category, route, year=year, page=0, subtype=subtype)
            else:
                yield self._listing_request(category, route)

    def _listing_request(
        self,
        category: str,
        route: str,
        *,
        year: int | None = None,
        page: int = 0,
        subtype: str | None = None,
    ) -> scrapy.Request:
        discovery = self.site_config.discovery
        tax_id = discovery.tax_id
        if tax_id is None:
            raise ValueError("cbic_taxinfo discovery requires tax_id")
        path = route.format(tax_id=tax_id)
        query: dict[str, str | int] = {}
        if category in self._paged_categories:
            if year is None:
                raise ValueError(f"{category} discovery requires a year")
            query = {
                "year": year,
                "page": page,
                "size": discovery.page_size,
                "taxId": tax_id,
            }
            if subtype:
                query["category"] = subtype
            if discovery.search_keyword:
                query["searchKeyword"] = discovery.search_keyword
        url = urljoin(str(self.site_config.base_url), path)
        if query:
            url = f"{url}?{urlencode(query)}"
        return self._request(
            url,
            callback=self.parse_listing,
            cb_kwargs={"category": category, "year": year, "page": page, "subtype": subtype},
            dont_filter=True,
        )

    def parse_listing(
        self,
        response: Response,
        category: str,
        year: int | None,
        page: int,
        subtype: str | None,
    ) -> Iterator[dict[str, object] | scrapy.Request]:
        if response.status >= 400:
            yield self._outcome(response, "blocked" if response.status in {403, 429} else "failed")
            return
        try:
            payload: object = json.loads(response.text)
        except (TypeError, ValueError):
            yield self._outcome(response, "incomplete", "Portal listing did not return JSON")
            return

        records = self._records(payload)
        if records is None:
            yield self._outcome(response, "incomplete", "Portal JSON did not contain a record list")
            return

        for record in records:
            if self._candidate_count >= self.site_config.discovery.max_documents:
                break
            if not self._is_required_tax(record):
                continue
            for candidate in self._record_candidates(category, record, response.url):
                yield candidate
                download_url = candidate.get("download_url")
                if self.site_config.parser.download_pdfs and isinstance(download_url, str):
                    yield self._request(
                        download_url,
                        callback=self.parse_download,
                        cb_kwargs={"candidate": candidate},
                    )

        discovery = self.site_config.discovery
        route = discovery.api_routes.get(category)
        next_page = page + 1
        if (
            route is not None
            and category in self._paged_categories
            and len(records) == discovery.page_size
            and next_page < discovery.max_pages
            and self._candidate_count < discovery.max_documents
        ):
            yield self._listing_request(category, route, year=year, page=next_page, subtype=subtype)

    @staticmethod
    def _records(payload: object) -> list[Mapping[str, Any]] | None:
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("content", "results", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [record for record in value if isinstance(record, Mapping)]
        return None

    def parse_download(
        self, response: Response, candidate: dict[str, object]
    ) -> Iterator[dict[str, object]]:
        if response.status >= 400:
            yield self._outcome(response, "blocked" if response.status in {403, 429} else "failed")
            return
        try:
            if response.body.startswith(b"%PDF-"):
                content = response.body
            else:
                payload = json.loads(response.body)
                encoded = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(encoded, str):
                    raise ValueError("Download response has no base64 data")
                content = base64.b64decode(encoded, validate=True)
            if not content.startswith(b"%PDF-"):
                raise ValueError("Download content has no PDF signature")
        except (ValueError, TypeError, binascii.Error) as exc:
            yield self._outcome(response, "incomplete", str(exc))
            return
        yield {
            "url": response.url,
            "body": content,
            "content_type": "application/pdf",
            "status": "success",
            "title": candidate.get("title", ""),
            "language": candidate.get("language", "en"),
            "document_id": candidate["document_id"],
            "discovered_metadata": {
                **cast(dict[str, object], candidate.get("metadata", {})),
                "category": candidate.get("category", ""),
                "tax_type": candidate.get("tax_type", "GST"),
                "listing_url": candidate.get("listing_url", ""),
            },
            "listing_url": candidate["listing_url"],
        }

    def _is_required_tax(self, record: Mapping[str, Any]) -> bool:
        expected_type = self.site_config.discovery.tax_type
        expected_id = self.site_config.discovery.tax_id
        nested_tax = record.get("tax") or record.get("taxId") or record.get("cbicTaxMst")
        if not isinstance(nested_tax, Mapping):
            return False
        actual_id = nested_tax.get("id")
        if expected_id is not None and actual_id != expected_id:
            return False
        if expected_type:
            names = [nested_tax.get(key) for key in ("taxName", "name", "taxType")]
            present_names = [str(value) for value in names if value]
            if present_names and all(expected_type.casefold() not in name.casefold() for name in present_names):
                return False
        return True

    def _record_candidates(
        self,
        category: str,
        record: Mapping[str, Any],
        listing_url: str,
    ) -> Iterator[dict[str, object]]:
        portal_id = record.get("id")
        if portal_id is None:
            return
        required_numbers = self.site_config.discovery.document_numbers
        number = str(record.get(self._number_fields.get(category, ""), ""))
        if required_numbers and number.casefold().lstrip("0") not in {
            value.casefold().lstrip("0") for value in required_numbers
        }:
            return
        title = self._first_text(record, self._title_fields.get(category, ("name",)))
        variants = self._language_variants(record)
        if not variants:
            variants = [(self.site_config.discovery.languages[0], "ENG")]

        for language, portal_language in variants:
            if language not in self.site_config.discovery.languages:
                continue
            record_key = f"{category}:{portal_id}:{language}"
            if record_key in self._seen_record_keys:
                continue
            if self._candidate_count >= self.site_config.discovery.max_documents:
                return
            self._seen_record_keys.add(record_key)

            source_url = self._source_url(category, portal_id, portal_language)
            download_url = self._download_url(category, portal_id, portal_language)
            self._candidate_count += 1
            yield {
                "schema_version": "1.0",
                "kind": "discovery_candidate",
                "source_id": source_id_for_url(source_url),
                "document_id": document_id_for_identifier("cbic-taxinfo", f"{category}:{portal_id}"),
                "record_id": record_key,
                "portal_record_id": str(portal_id),
                "content_id": str(record.get("contentId", "")),
                "observed_url": source_url,
                "canonical_url": source_url,
                "listing_url": listing_url,
                "site": self.site_config.name,
                "category": category,
                "tax_type": self.site_config.discovery.tax_type or "",
                "language": language,
                "title": title,
                "download_url": download_url,
                "metadata": {
                    "number": record.get(self._number_fields.get(category, "")),
                    "date": self._first_value(record, ("notificationDt", "circularDt", "instructionDt", "orderDt")),
                    "subtype": self._first_value(
                        record,
                        ("notificationCategory", "circularCategory", "instructionCategory", "orderCategory"),
                    ),
                    "portal_language": portal_language,
                },
                "raw_record": dict(record),
            }

    def _source_url(self, category: str, portal_id: object, portal_language: str) -> str:
        if category in self._tab_names:
            path = f"view-pdf/{portal_id}/{portal_language}/{self._tab_names[category]}"
        else:
            singular = "act" if category == "acts" else "rule"
            path = f"api/cbic-{singular}-msts/{portal_id}"
        return urljoin(str(self.site_config.base_url), path)

    def _download_url(self, category: str, portal_id: object, portal_language: str) -> str | None:
        route = self.site_config.discovery.download_routes.get(category)
        if route is None:
            return None
        return urljoin(
            str(self.site_config.base_url),
            route.format(id=portal_id, language=portal_language),
        )

    @staticmethod
    def _language_variants(record: Mapping[str, Any]) -> list[tuple[str, str]]:
        variants: list[tuple[str, str]] = []
        if record.get("docFilePath") or record.get("doc_file_path") or record.get("contentFilePath"):
            variants.append(("en", "ENG"))
        if record.get("docFilePathHi") or record.get("doc_file_path_hi") or record.get("contentFilePathHi"):
            variants.append(("hi", "HINDI"))
        return variants

    @staticmethod
    def _first_text(record: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        value = CbicTaxInfoSpider._first_value(record, keys)
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _first_value(record: Mapping[str, Any], keys: tuple[str, ...]) -> object | None:
        return next((record[key] for key in keys if record.get(key) not in (None, "")), None)

    @staticmethod
    def _outcome(response: Response, status: str, error: str | None = None) -> dict[str, object]:
        return {
            "url": response.url,
            "status": status,
            "http_status": response.status,
            "error": error or f"HTTP {response.status}",
        }

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, cast

from nagrik_ai.config.config_models import SiteConfig
from nagrik_ai.crawler.crawl_manifest import CrawlManifest, FetchStatus, ManifestRecord
from nagrik_ai.crawler.url_utils import source_id_for_url
from nagrik_ai.models.document import DiscoveryCandidate
from nagrik_ai.parser.document_pipeline import mark_previous_normalized_version_historical, normalize_source

logger = logging.getLogger(__name__)


class PipelineSpider(Protocol):
    output_dir: Path
    site_config: SiteConfig


class DocumentSavePipeline:
    def __init__(self) -> None:
        self.items: list[dict[str, object]] = []
        self.manifest: CrawlManifest | None = None

    def open_spider(self, spider: object) -> None:
        typed_spider = cast(PipelineSpider, spider)
        self.manifest = CrawlManifest.for_crawled_dir(
            typed_spider.output_dir,
            site=typed_spider.site_config.name,
        )

    def process_item(self, item: dict[str, object], spider: object | None = None) -> dict[str, object]:
        if item.get("kind") == "discovery_candidate":
            candidate = DiscoveryCandidate.model_validate(item)
            if candidate.source_id != source_id_for_url(candidate.canonical_url):
                raise ValueError("Discovery candidate source_id does not match canonical_url")
            if self.manifest is None:
                if spider is None:
                    raise RuntimeError("DocumentSavePipeline requires a spider")
                self.open_spider(spider)
            manifest = self.manifest
            if manifest is None:
                raise RuntimeError("DocumentSavePipeline could not initialize a manifest")
            raw = (json.dumps(candidate.raw_record, sort_keys=True, ensure_ascii=False) + "\n").encode()
            candidate_version, created = manifest.store_response(
                url=candidate.canonical_url,
                content=raw,
                content_type="application/json",
                title=candidate.title,
                fetch_time=candidate.discovered_at.isoformat(),
            )
            manifest.resolve_document_identity(
                source_id=candidate_version.source_id,
                document_id=candidate.document_id,
                method="portal_record_id",
                evidence={"record_id": candidate.portal_record_id, "category": candidate.category},
                confidence=1.0,
            )
            candidate_version = manifest.state.versions[candidate_version.version_id]
            discovery_metadata = dict(candidate.metadata)
            discovery_metadata.update(
                {
                    "category": candidate.category,
                    "tax_type": candidate.tax_type,
                    "portal_record_id": candidate.portal_record_id,
                    "record_id": candidate.record_id,
                    "listing_url": candidate.listing_url,
                }
            )
            normalized, normalized_path = normalize_source(
                manifest,
                candidate_version,
                raw,
                document_id=candidate.document_id,
                language=candidate.language,
                discovered_metadata=discovery_metadata,
                official_file_url=candidate.download_url,
                provenance_url=candidate.listing_url,
                provenance_method="portal_json_discovery",
            )
            if created:
                mark_previous_normalized_version_historical(manifest, candidate_version)
            item["url"] = candidate.canonical_url
            item["status"] = "success" if created else "unchanged"
            item["version_id"] = candidate_version.version_id
            item["content_hash"] = candidate_version.content_hash
            item["local_path"] = candidate_version.local_path
            item["normalized_path"] = normalized_path.relative_to(manifest.site_root).as_posix()
            item["normalized_document"] = normalized.model_dump(mode="json")
            self.items.append(item)
            return item

        if self.manifest is None:
            if spider is None:
                raise RuntimeError("DocumentSavePipeline requires a spider")
            self.open_spider(spider)
        manifest = self.manifest
        if manifest is None:
            raise RuntimeError("DocumentSavePipeline could not initialize a manifest")

        url = str(item["url"])
        status = str(item.get("status", "success"))
        version: ManifestRecord | None = None
        if status == "success":
            body = item.get("body")
            if not isinstance(body, bytes):
                raise TypeError("Successful crawl items must include byte content")
            version, created = manifest.store_response(
                url=url,
                content=body,
                content_type=str(item.get("content_type", "text/html")),
                title=str(item.get("title", "")),
            )
            item["status"] = "success" if created else "unchanged"
            document_id = item.get("document_id")
            if isinstance(document_id, str) and document_id:
                manifest.resolve_document_identity(
                    source_id=version.source_id,
                    document_id=document_id,
                    method="portal_download_record",
                    evidence={"download_url": url, "listing_url": item.get("listing_url", "")},
                    confidence=1.0,
                )
            try:
                normalized, normalized_path = normalize_source(
                    manifest,
                    version,
                    body,
                    language=str(item.get("language", "en")),
                    discovered_metadata=cast(
                        dict[str, str | int | float | bool | None], item.get("discovered_metadata", {})
                    ),
                )
                if created:
                    mark_previous_normalized_version_historical(manifest, version)
                item["normalized_path"] = normalized_path.relative_to(manifest.site_root).as_posix()
                item["normalized_document"] = normalized.model_dump(mode="json")
                if normalized.extraction.quality_status in {
                    "unsupported", "failed", "incomplete", "ocr_required", "low_text"
                }:
                    item["status"] = "incomplete"
                    raw_http_status = item.get("http_status")
                    manifest.record_outcome(
                        url=url,
                        status="incomplete",
                        http_status=raw_http_status if isinstance(raw_http_status, int) else None,
                        error=f"Extraction incomplete: {normalized.extraction.quality_status}",
                    )
            except Exception as exc:
                logger.exception("Could not normalize %s", url)
                item["status"] = "incomplete"
                item["error"] = str(exc)
                manifest.record_outcome(url=url, status="incomplete", error=str(exc))
        elif status == "reused":
            version = manifest.mark_reused(url)
            if version is None:
                raise FileNotFoundError(f"Managed crawl artifact is unavailable for {url}")
        else:
            allowed_statuses = {"blocked", "failed", "incomplete"}
            if status not in allowed_statuses:
                raise ValueError(f"Unsupported crawl outcome status: {status}")
            raw_http_status = item.get("http_status")
            http_status = raw_http_status if isinstance(raw_http_status, int) else None
            manifest.record_outcome(
                url=url,
                status=cast(FetchStatus, status),
                http_status=http_status,
                error=str(item["error"]) if item.get("error") else None,
            )

        if version is not None:
            item["source_id"] = version.source_id
            item["version_id"] = version.version_id
            item["content_hash"] = version.content_hash
            item["local_path"] = version.local_path
        self.items.append(item)
        return item

    def close_spider(self, spider: object) -> None:
        saved = sum(1 for item in self.items if item.get("status") == "success")
        unchanged = sum(1 for item in self.items if item.get("status") in {"unchanged", "reused"})
        failed = sum(1 for item in self.items if item.get("status") in {"blocked", "failed", "incomplete"})
        logger.info("Spider finished: %d saved, %d unchanged/reused, %d failed", saved, unchanged, failed)

        output_dir: Path | None = getattr(spider, "output_dir", None)
        if output_dir and self.manifest is not None:
            mappings: dict[str, dict[str, str]] = {}
            for version in self.manifest.state.versions.values():
                local_path = Path(version.local_path)
                try:
                    filename = local_path.relative_to("crawled").as_posix()
                except ValueError:
                    filename = local_path.as_posix()
                mappings[filename] = {"url": version.observed_source_url, "title": version.title}
            mapping_file = output_dir / "url_mappings.json"
            CrawlManifest._atomic_write(mapping_file, (json.dumps(mappings, indent=2, sort_keys=True) + "\n").encode())
            logger.info("Wrote URL mappings to %s", mapping_file)

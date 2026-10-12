from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, cast

from nagrik_ai.config.config_models import SiteConfig
from nagrik_ai.crawler.crawl_manifest import CrawlManifest, FetchStatus, ManifestRecord

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
        # Phase 5 emits metadata-only candidates. Persistence is deliberately
        # deferred until the normalized document contract is introduced.
        if item.get("kind") == "discovery_candidate":
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

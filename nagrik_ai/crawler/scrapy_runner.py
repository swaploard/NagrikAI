from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scrapy.crawler import CrawlerProcess

from nagrik_ai.config.config_models import SiteConfig
from nagrik_ai.crawler.crawl_manifest import CrawlManifest
from nagrik_ai.crawler.spiders.cbic_taxinfo_spider import CbicTaxInfoSpider
from nagrik_ai.crawler.spiders.site_spider import SiteSpider

logger = logging.getLogger(__name__)


def spider_settings(
    site_config: SiteConfig,
    max_depth: int | None = None,
    page_limit: int | None = None,
    document_limit: int | None = None,
) -> dict[str, Any]:
    crawler = site_config.crawler
    depth = crawler.max_depth if max_depth is None else max_depth
    settings = {
        "USER_AGENT": crawler.user_agent,
        "CONCURRENT_REQUESTS": crawler.max_concurrency,
        "CONCURRENT_REQUESTS_PER_DOMAIN": crawler.per_domain_concurrency,
        "DOWNLOAD_DELAY": crawler.request_delay,
        "DOWNLOAD_TIMEOUT": crawler.timeout,
        "DEPTH_LIMIT": depth if depth > 0 else 0,
        "ROBOTSTXT_OBEY": True,
        "AUTOTHROTTLE_ENABLED": crawler.autothrottle_enabled,
        "AUTOTHROTTLE_START_DELAY": crawler.autothrottle_start_delay,
        "AUTOTHROTTLE_MAX_DELAY": crawler.autothrottle_max_delay,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": crawler.autothrottle_target_concurrency,
        "HTTPERROR_ALLOW_ALL": True,
    }
    if page_limit is not None:
        settings["CLOSESPIDER_PAGECOUNT"] = page_limit
    if document_limit is not None:
        settings["CLOSESPIDER_ITEMCOUNT"] = document_limit
    return settings


def site_spider_class(
    site_config: SiteConfig,
    max_depth: int | None = None,
    page_limit: int | None = None,
    document_limit: int | None = None,
) -> type[SiteSpider]:
    safe_name = "".join(character if character.isalnum() else "_" for character in site_config.name)
    attributes: dict[str, Any] = {
        "name": f"site_spider_{safe_name}",
        "custom_settings": spider_settings(
            site_config,
            max_depth=max_depth,
            page_limit=page_limit,
            document_limit=document_limit,
        ),
    }
    spider_base = CbicTaxInfoSpider if site_config.name == "cbic_taxinfo" else SiteSpider
    return type(f"{safe_name.title().replace('_', '')}SiteSpider", (spider_base,), attributes)


def run_scrapy_crawl(
    site_config: SiteConfig,
    output_dir: Path,
    max_depth: int | None = None,
    manage: bool = False,
    force: bool = False,
    incremental: bool = False,
    page_limit: int | None = None,
    document_limit: int | None = None,
) -> int:
    return run_batch_scrapy_crawl(
        [(site_config, output_dir, manage)],
        max_depth=max_depth,
        force=force,
        incremental=incremental,
        page_limit=page_limit,
        document_limit=document_limit,
    )


def run_batch_scrapy_crawl(
    jobs: list[tuple[SiteConfig, Path, bool]],
    max_depth: int | None = None,
    force: bool = False,
    incremental: bool = False,
    page_limit: int | None = None,
    document_limit: int | None = None,
) -> int:
    if not jobs:
        return 0
    logging.getLogger("scrapy.core.scraper").setLevel(logging.INFO)

    settings = {
        "BOT_NAME": "nagrik_ai",
        "TELNETCONSOLE_ENABLED": False,
        "ROBOTSTXT_OBEY": True,
        "LOG_LEVEL": "INFO",
        "LOG_FORMATTER": "scrapy.logformatter.LogFormatter",
        "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "DOWNLOAD_HANDLERS": {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "PLAYWRIGHT_MAX_CONTEXTS": 2,
        "PLAYWRIGHT_MAX_PAGES_PER_CONTEXT": 4,
        "ITEM_PIPELINES": {
            "nagrik_ai.crawler.pipelines.DocumentSavePipeline": 300,
        },
    }

    process = CrawlerProcess(settings)

    for site_config, output_dir, manage in jobs:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_record = {
            "started_at": datetime.now(UTC).isoformat(),
            "site": site_config.name,
            "page_cap": page_limit,
            "document_cap": document_limit,
            "depth": site_config.crawler.max_depth if max_depth is None else max_depth,
            "site_config": site_config.model_dump(mode="json"),
        }
        CrawlManifest._atomic_write(
            output_dir.parent / "manifests" / "last_crawl_run.json",
            (json.dumps(run_record, sort_keys=True, indent=2) + "\n").encode(),
        )
        spider_class = site_spider_class(
            site_config,
            max_depth=max_depth,
            page_limit=page_limit,
            document_limit=document_limit,
        )
        process.crawl(
            spider_class,
            site_config=site_config,
            output_dir=output_dir,
            manage=manage,
            force=force,
            incremental=incremental,
        )

    process.start()
    return 0

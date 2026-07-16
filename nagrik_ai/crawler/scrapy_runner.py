from __future__ import annotations

import logging
from pathlib import Path

from scrapy.crawler import CrawlerProcess

from nagrik_ai.config.config_models import SiteConfig
from nagrik_ai.crawler.spiders.site_spider import SiteSpider

logger = logging.getLogger(__name__)


def run_scrapy_crawl(
    site_config: SiteConfig,
    output_dir: Path,
    max_depth: int | None = None,
    manage: bool = False,
) -> int:
    return run_batch_scrapy_crawl(
        [(site_config, output_dir, manage)],
        max_depth=max_depth,
    )


def run_batch_scrapy_crawl(
    jobs: list[tuple[SiteConfig, Path, bool]],
    max_depth: int | None = None,
) -> int:
    if not jobs:
        return 0

    first = jobs[0][0]
    depth = max_depth if max_depth is not None else first.crawler.max_depth

    settings = {
        "BOT_NAME": "nagrik_ai",
        "USER_AGENT": first.crawler.user_agent,
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": first.crawler.max_concurrency,
        "DOWNLOAD_DELAY": first.crawler.request_delay,
        "DOWNLOAD_TIMEOUT": first.crawler.timeout,
        "DEPTH_LIMIT": depth if depth > 0 else 0,
        "LOG_LEVEL": "INFO",
        "LOG_FORMATTER": "scrapy.logformatter.LogFormatter",
        "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "ITEM_PIPELINES": {
            "nagrik_ai.crawler.pipelines.DocumentSavePipeline": 300,
        },
        "AUTOTHROTTLE_ENABLED": False,
    }

    process = CrawlerProcess(settings)

    for site_config, output_dir, manage in jobs:
        output_dir.mkdir(parents=True, exist_ok=True)
        process.crawl(
            SiteSpider,
            site_config=site_config,
            output_dir=output_dir,
            manage=manage,
        )

    process.start()
    return 0

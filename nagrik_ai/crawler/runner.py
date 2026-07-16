from __future__ import annotations

from pathlib import Path

from nagrik_ai.config.config_models import SiteConfig
from nagrik_ai.crawler.scrapy_runner import run_scrapy_crawl


def run_crawl(
    site_config: SiteConfig,
    output_dir: Path,
    max_depth: int | None = None,
    manage: bool = False,
) -> int:
    crawled_dir = output_dir / site_config.name / "crawled"
    return run_scrapy_crawl(site_config, crawled_dir, max_depth=max_depth, manage=manage)

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import scrapy
from scrapy.http import Response
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from nagrik_ai.config.config_models import SiteConfig
from nagrik_ai.models.document import Document


def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return "index"
    return path


class SiteSpider(CrawlSpider):
    name = "site_spider"

    def __init__(
        self,
        site_config: SiteConfig,
        output_dir: Path,
        manage: bool = False,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.site_config = site_config
        self.output_dir = output_dir
        self.manage = manage

        self.allowed_domains = list(site_config.allowed_domains)
        self.start_urls = [str(u) for u in site_config.start_urls]

        self.rules = (
            Rule(
                LinkExtractor(allow_domains=self.allowed_domains),
                callback="parse_item",
                follow=True,
            ),
        )

        super().__init__(*args, **kwargs)

    def parse_start_url(
        self,
        response: Response,
        **_kwargs: object,
    ) -> Iterator[dict[str, object] | scrapy.Request]:
        return self.parse_item(response)

    def parse_item(self, response: Response) -> Iterator[dict[str, object] | scrapy.Request]:
        url = response.url
        filename = url_to_filename(url)
        filepath = (self.output_dir / filename).with_suffix(".html")

        if self.manage and filepath.exists():
            yield {"url": url, "title": "", "doc_id": "", "skipped": True}
        else:
            title = response.css("title::text").get(default="").strip()

            docs = Document(
                content=response.text,
                source="crawl",
                site=self.site_config.name,
                title=title,
                url=url,
            )

            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(docs.content, encoding="utf-8")

            yield {
                "url": url,
                "title": title,
                "doc_id": docs.doc_id,
                "filename": f"{filename}.html",
                "skipped": False,
            }

        yield from self._spa_requests(response)

    def _spa_requests(self, response: Response) -> Iterator[scrapy.Request]:
        for href in response.css('a[href*="#t="]::attr(href)').getall():
            href = href.strip()
            if href.startswith("//"):
                href = "https:" + href
            if "#t=" not in href:
                continue

            before_hash = href.split("#t=", 1)[0]
            target_file = href.split("#t=", 1)[1].split("#")[0]
            if not target_file:
                continue

            base_url = before_hash if before_hash else response.url
            parsed = urlparse(base_url)
            parent = parsed.path.rsplit("/", 1)[0] if "/" in parsed.path else ""
            new_path = f"{parent}/{target_file}" if parent else target_file
            actual_url = urlunparse(parsed._replace(path=new_path, fragment="", query=""))

            if any(d in actual_url for d in self.allowed_domains):
                yield scrapy.Request(url=actual_url, callback=self.parse_item)

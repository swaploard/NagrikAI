from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse, urlunparse

import scrapy
from scrapy.http import Response
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from twisted.python.failure import Failure

from nagrik_ai.config.config_models import SiteConfig
from nagrik_ai.crawler.crawl_manifest import CrawlManifest


class RequestFailure(Protocol):
    request: scrapy.Request


class SiteSpider(CrawlSpider):
    name = "site_spider"
    handle_httpstatus_all = True

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
        self.site_config = site_config
        self.output_dir = output_dir
        self.manage = manage
        self.force = force
        self.incremental = incremental
        self.manifest = CrawlManifest.for_crawled_dir(output_dir, site=site_config.name)

        self.allowed_domains = list(site_config.allowed_domains)
        self.start_urls = [str(u) for u in site_config.start_urls]

        self.rules = (
            Rule(
                LinkExtractor(allow_domains=self.allowed_domains),
                callback="parse_item",
                follow=True,
                errback="handle_request_error",
            ),
        )

        super().__init__(*args, **kwargs)

    def start_requests(self) -> Iterator[scrapy.Request]:
        for url in self.start_urls:
            yield scrapy.Request(url=url, dont_filter=True, errback=self.handle_request_error)

    def parse_start_url(
        self,
        response: Response,
        **_kwargs: object,
    ) -> Iterator[dict[str, object] | scrapy.Request]:
        return self.parse_item(response)

    def parse_item(self, response: Response) -> Iterator[dict[str, object] | scrapy.Request]:
        url = response.url
        if response.status >= 400:
            yield {
                "url": url,
                "status": "blocked" if response.status in {403, 429} else "failed",
                "http_status": response.status,
                "error": f"HTTP {response.status}",
            }
            return

        reusable = self.manifest.reusable_version(url) if self.manage and not self.force else None
        if reusable is not None:
            yield {
                "url": url,
                "title": reusable.title,
                "status": "reused",
                "version_id": reusable.version_id,
                "local_path": reusable.local_path,
            }
        else:
            title = response.css("title::text").get(default="").strip()
            content_type_header = response.headers.get(b"Content-Type")
            content_type = content_type_header.decode("latin-1") if content_type_header is not None else "text/html"
            yield {
                "url": url,
                "title": title,
                "body": response.body,
                "content_type": content_type,
                "status": "success",
                "http_status": response.status,
            }

        yield from self._spa_requests(response)

    def handle_request_error(self, failure: Failure) -> Iterator[dict[str, object]]:
        request = cast(RequestFailure, failure).request
        message = failure.getErrorMessage()
        lowered = message.casefold()
        status = "blocked" if "robot" in lowered or "403" in lowered or "429" in lowered else "failed"
        response = getattr(failure.value, "response", None)
        http_status = getattr(response, "status", None)
        if http_status in {403, 429}:
            status = "blocked"
        yield {
            "url": request.url,
            "status": status,
            "http_status": http_status,
            "error": message,
        }

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

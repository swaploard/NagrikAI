from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urlparse, urlunparse

import scrapy
from scrapy.http import Response, TextResponse
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
        allow_patterns = site_config.discovery.allowed_path_patterns or ()

        self.rules = (
            Rule(
                LinkExtractor(allow_domains=self.allowed_domains, allow=allow_patterns),
                callback="parse_item",
                follow=site_config.discovery.follow_links,
                errback="handle_request_error",
                process_request="prepare_request",
            ),
        )

        super().__init__(*args, **kwargs)

    def start_requests(self) -> Iterator[scrapy.Request]:
        for url in self.start_urls:
            if self._is_allowed_url(url):
                yield self._request(url, dont_filter=True)

    async def start(self) -> AsyncIterator[scrapy.Request]:
        for request in self.start_requests():
            yield request

    def prepare_request(self, request: scrapy.Request, _response: Response) -> scrapy.Request | None:
        if not self._is_allowed_url(request.url):
            return None
        request.errback = self.handle_request_error
        request.meta.update(self._request_meta(request.url))
        return request

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

        if self._is_incomplete_shell(response):
            yield {
                "url": url,
                "status": "incomplete",
                "http_status": response.status,
                "error": "JavaScript shell did not reach the configured completion condition",
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
            title = response.css("title::text").get(default="").strip() if isinstance(response, TextResponse) else ""
            content_type_header = response.headers.get(b"Content-Type")
            content_type = content_type_header.decode("latin-1") if content_type_header is not None else "text/html"
            yield {
                "url": url,
                "title": title,
                "body": response.body,
                "content_type": content_type,
                "status": "success",
                "http_status": response.status,
                "language": self._language_for_url(url),
            }

        if isinstance(response, TextResponse):
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
                yield self._request(actual_url, callback=self.parse_item)

    def _request(
        self,
        url: str,
        *,
        callback: Any = None,
        cb_kwargs: dict[str, Any] | None = None,
        dont_filter: bool = False,
    ) -> scrapy.Request:
        headers = self.site_config.discovery.public_headers
        return scrapy.Request(
            url=url,
            callback=callback,
            errback=self.handle_request_error,
            headers=headers,
            meta=self._request_meta(url),
            cb_kwargs=cb_kwargs,
            dont_filter=dont_filter,
        )

    def _request_meta(self, url: str) -> dict[str, object]:
        strategy = self.site_config.crawler.crawl_strategy
        patterns = self.site_config.discovery.browser_path_patterns
        needs_browser = strategy == "browser" or (
            strategy == "hybrid" and any(re.search(pattern, urlparse(url).path) for pattern in patterns)
        )
        if not needs_browser:
            return {}

        meta: dict[str, object] = {
            "playwright": True,
            # Pages are closed by scrapy-playwright because they are never included
            # in callbacks. This also covers HTTP and callback error paths.
            "playwright_include_page": False,
        }
        selector = self.site_config.discovery.readiness_selector
        if selector:
            from scrapy_playwright.page import PageMethod  # type: ignore[import-untyped]

            meta["playwright_page_methods"] = [PageMethod("wait_for_selector", selector)]
        elif self.site_config.discovery.completion_condition == "network_idle":
            meta["playwright_page_goto_kwargs"] = {"wait_until": "networkidle"}
        return meta

    def _is_allowed_url(self, url: str) -> bool:
        patterns = self.site_config.discovery.allowed_path_patterns
        if not patterns:
            return True
        path = urlparse(url).path
        return any(re.search(pattern, path) for pattern in patterns)

    def _is_incomplete_shell(self, response: Response) -> bool:
        if not isinstance(response, TextResponse):
            return False
        text = response.text.casefold()
        shell_markers = ("app-loading", "enable javascript", "javascript is required")
        if not any(marker in text for marker in shell_markers):
            return False
        selector = self.site_config.discovery.readiness_selector
        if response.meta.get("playwright") and selector:
            return not bool(response.css(selector))
        return True

    def _language_for_url(self, url: str) -> str:
        path = urlparse(url).path.casefold()
        if "/hindi/" in path or "/hi/" in path:
            return "hi"
        return self.site_config.discovery.languages[0] if self.site_config.discovery.languages else "en"

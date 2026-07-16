from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

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

    def parse_start_url(self, response: Response, **_kwargs: object) -> dict[str, object]:
        return self.parse_item(response)

    def parse_item(self, response: Response) -> dict[str, object]:
        url = response.url
        filename = url_to_filename(url)
        filepath = (self.output_dir / filename).with_suffix(".html")

        if self.manage and filepath.exists():
            return {"url": url, "title": "", "doc_id": "", "skipped": True}

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

        return {"url": url, "title": title, "doc_id": docs.doc_id, "filename": f"{filename}.html", "skipped": False}

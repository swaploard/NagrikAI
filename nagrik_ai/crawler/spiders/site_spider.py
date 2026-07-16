from __future__ import annotations

from pathlib import Path

from scrapy.http import Response
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from nagrik_ai.config.config_models import SiteConfig
from nagrik_ai.models.document import Document


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

    def parse_start_url(self, response: Response) -> dict[str, object]:
        return self.parse_item(response)

    def parse_item(self, response: Response) -> dict[str, object]:
        url = response.url

        if self.manage:
            existing = (self.output_dir / url.replace("/", "_").replace(":", "_")).with_suffix(".md")
            if existing.exists():
                return {"url": url, "title": "", "doc_id": "", "skipped": True}

        title = response.css("title::text").get(default="").strip()

        body = response.css("body")
        for sel in self.site_config.parser.strip_selectors:
            for element in body.css(sel):
                element.drop()
        content = " ".join(text.strip() for text in body.xpath(".//text()").getall() if text.strip())
        content = "\n\n".join(line.strip() for line in content.split("  ") if line.strip())

        doc = Document(
            content=content,
            source="crawl",
            site=self.site_config.name,
            title=title,
            url=url,
        )

        path = self.output_dir / f"{doc.doc_id}.md"
        path.write_text(doc.content, encoding="utf-8")

        return {"url": url, "title": title, "doc_id": doc.doc_id, "skipped": False}

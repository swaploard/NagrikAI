from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from nagrik_ai.config.config_models import SiteConfig
from nagrik_ai.models.document import Document


class BaseCrawler:
    def __init__(self, site_config: SiteConfig) -> None:
        self.site_config = site_config
        self.cfg = site_config.crawler
        self._visited: set[str] = set()

    async def crawl(self) -> AsyncIterator[Document]:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.cfg.timeout),
            headers={"User-Agent": self.cfg.user_agent},
        ) as client:
            sem = asyncio.Semaphore(self.cfg.max_concurrency)
            to_visit = [str(u) for u in self.site_config.start_urls]

            while to_visit:
                batch = to_visit[: self.cfg.max_concurrency]
                to_visit = to_visit[self.cfg.max_concurrency :]

                tasks = [self._fetch_and_parse(client, url, sem) for url in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, BaseException):
                        continue
                    doc, links = result
                    self._visited.add(doc.url)
                    yield doc
                    for link in links:
                        if link not in self._visited and link not in to_visit:
                            to_visit.append(link)

                await asyncio.sleep(self.cfg.request_delay)

    async def _fetch_and_parse(
        self, client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore
    ) -> tuple[Document, list[str]]:
        async with sem:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string if soup.title and soup.title.string else ""
            content = soup.get_text(separator="\n", strip=True)
            links = self._extract_links(soup, url)

            doc = Document(
                content=content,
                source="crawl",
                site=self.site_config.name,
                title=title,
                url=url,
                crawled_at=datetime.now(),
            )
            return doc, links

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        links: list[str] = []
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, str(a["href"]))
            parsed = urlparse(href)
            if parsed.netloc in self.site_config.allowed_domains and parsed.scheme in ("http", "https"):
                links.append(href)
        return links

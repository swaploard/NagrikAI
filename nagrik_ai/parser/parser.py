from __future__ import annotations

from datetime import datetime
from pathlib import Path

import html2text
from bs4 import BeautifulSoup

from nagrik_ai.models.document import Document

YAML_TEMPLATE = """---
source: {source}
site: {site}
title: {title}
url: {url}
crawled_at: {crawled_at}
---
"""


class HTMLParser:
    def __init__(self) -> None:
        self._converter = html2text.HTML2Text()
        self._converter.body_width = 0
        self._converter.ignore_links = False
        self._converter.ignore_images = True
        self._converter.ignore_emphasis = False

    def parse_html(self, html: str, url: str = "", site: str = "", title: str = "") -> Document:
        soup = BeautifulSoup(html, "html.parser")
        text_content = soup.get_text(separator="\n", strip=True)
        markdown = self._converter.handle(text_content)
        frontmatter = YAML_TEMPLATE.format(
            source="parsed",
            site=site,
            title=title,
            url=url,
            crawled_at=datetime.now().isoformat(),
        )
        content = frontmatter + markdown
        return Document(
            content=content,
            source="parsed",
            site=site,
            title=title,
            url=url,
        )

    def parse_file(self, html_path: Path, site: str = "") -> Document:
        html = html_path.read_text(encoding="utf-8")
        return self.parse_html(html, url="", site=site)

    def save_parsed(self, doc: Document, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{doc.doc_id}.md"
        path.write_text(doc.content, encoding="utf-8")
        return path

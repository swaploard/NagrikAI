from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4


class Document:
    def __init__(
        self,
        *,
        content: str,
        source: str,
        site: str,
        title: str = "",
        url: str = "",
        crawled_at: datetime | None = None,
        doc_id: str | None = None,
    ) -> None:
        self.doc_id = doc_id or uuid4().hex
        self.content = content
        self.source = source
        self.site = site
        self.title = title
        self.url = url
        self.crawled_at = crawled_at or datetime.now()

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "source": self.source,
            "site": self.site,
            "title": self.title,
            "url": self.url,
            "crawled_at": self.crawled_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        return cls(
            content=data["content"],
            source=data["source"],
            site=data["site"],
            title=data.get("title", ""),
            url=data.get("url", ""),
            crawled_at=datetime.fromisoformat(data["crawled_at"]) if data.get("crawled_at") else None,
            doc_id=data.get("doc_id"),
        )

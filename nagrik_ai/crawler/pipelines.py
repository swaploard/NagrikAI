from __future__ import annotations

import logging

from scrapy import Spider

logger = logging.getLogger(__name__)


class DocumentSavePipeline:
    def __init__(self) -> None:
        self.items: list[dict[str, object]] = []

    def process_item(self, item: dict[str, object], _spider: Spider) -> dict[str, object]:
        if item.get("skipped"):
            return item
        self.items.append(item)
        return item

    def close_spider(self, _spider: Spider) -> None:
        saved = sum(1 for it in self.items if not it.get("skipped", True))
        skipped = sum(1 for it in self.items if it.get("skipped"))
        logger.info("Spider finished: %d saved, %d skipped", saved, skipped)

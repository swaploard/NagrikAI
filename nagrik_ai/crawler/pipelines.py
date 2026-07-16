from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentSavePipeline:
    def __init__(self) -> None:
        self.items: list[dict[str, object]] = []

    def process_item(self, item: dict[str, object], _spider: object | None = None) -> dict[str, object]:
        if item.get("skipped"):
            return item
        self.items.append(item)
        return item

    def close_spider(self, spider: object) -> None:
        saved = sum(1 for it in self.items if not it.get("skipped", True))
        skipped = sum(1 for it in self.items if it.get("skipped"))
        logger.info("Spider finished: %d saved, %d skipped", saved, skipped)

        output_dir: Path | None = getattr(spider, "output_dir", None)
        if output_dir and saved > 0:
            mappings: dict[str, dict[str, str]] = {}
            for item in self.items:
                filename = str(item.get("filename", ""))
                if filename:
                    mappings[filename] = {
                        "url": str(item.get("url", "")),
                        "title": str(item.get("title", "")),
                    }
            mapping_file = output_dir / "url_mappings.json"
            mapping_file.write_text(json.dumps(mappings, indent=2), encoding="utf-8")
            logger.info("Wrote URL mappings to %s", mapping_file)

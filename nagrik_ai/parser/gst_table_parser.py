from __future__ import annotations

import hashlib
import re
from typing import Any

from lxml import html

from nagrik_ai.models.document import TablePayload, TableRow


class GSTTableParser:
    parser_name = "GSTTableParser"
    parser_version = "1.0"

    def parse_html(self, html_content: str) -> list[TablePayload]:
        tree: Any = html.fromstring(html_content)
        tables: list[TablePayload] = []
        for table_index, table in enumerate(tree.xpath("//table")):
            rows = table.xpath(".//tr")
            if not rows:
                continue
            raw_rows = [[self._cell_text(cell) for cell in row.xpath("./th|./td")] for row in rows]
            headers = self._headers(raw_rows[0])
            heading, schedule = self._context(table)
            parsed_rows: list[TableRow] = []
            footnotes: list[str] = []
            for row_index, values in enumerate(raw_rows[1:]):
                if len(values) == 1 and self._is_footnote(values[0]):
                    footnotes.append(values[0])
                    continue
                cells = {
                    header: values[index] if index < len(values) else ""
                    for index, header in enumerate(headers)
                }
                condition = self._field(cells, "condition")
                parsed_rows.append(
                    TableRow(
                        row_index=row_index,
                        cells=cells,
                        heading=heading,
                        schedule=schedule,
                        condition=condition,
                        footnotes=list(footnotes),
                    )
                )
            if footnotes:
                parsed_rows = [row.model_copy(update={"footnotes": list(footnotes)}) for row in parsed_rows]
            identity = f"{table_index}:{headers}:{raw_rows}".encode()
            tables.append(
                TablePayload(
                    table_id=hashlib.sha256(identity).hexdigest(),
                    headers=headers,
                    rows=parsed_rows,
                    title=heading,
                    footnotes=footnotes,
                )
            )
        return tables

    @staticmethod
    def _cell_text(cell: Any) -> str:
        return " ".join(cell.text_content().split())

    @staticmethod
    def _headers(values: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        result: list[str] = []
        for index, value in enumerate(values):
            base = value.strip() or f"column_{index + 1}"
            seen[base] = seen.get(base, 0) + 1
            result.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
        return result

    @staticmethod
    def _context(table: Any) -> tuple[str | None, str | None]:
        heading_nodes = table.xpath("preceding::*[self::h1 or self::h2 or self::h3 or self::h4][1]")
        heading = " ".join(heading_nodes[0].text_content().split()) if heading_nodes else None
        schedule = heading if heading and "schedule" in heading.casefold() else None
        return heading, schedule

    @staticmethod
    def _field(cells: dict[str, str], name: str) -> str | None:
        return next((value for key, value in cells.items() if name in key.casefold() and value), None)

    @staticmethod
    def _is_footnote(value: str) -> bool:
        return bool(re.match(r"^(?:note|footnote|\*+)", value.strip(), re.I))

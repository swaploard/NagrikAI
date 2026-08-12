"""HTML content parser module.

This module contains the Parser class, which loads site-specific
configurations and extracts content from HTML pages accordingly.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import html2text
import yaml
from lxml import html

from nagrik_ai.config.config_models import SiteConfig

MIN_NESTED_PATH_PARTS = 2


class DirectoryScope:
    """Context manager for temporarily changing directory context without modifying instance state."""

    def __init__(self, original_path: str, scoped_path: str) -> None:
        self.original_path = original_path
        self.scoped_path = scoped_path

    def __enter__(self) -> str:
        return self.scoped_path

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        pass


class Parser:
    """HTML content parser that uses site-specific configurations."""

    def __init__(
        self,
        site_name: str,
        site_config: SiteConfig,
        input_dir: str,
        output_dir: str,
    ) -> None:
        self.setup_logging()

        self.site = site_name
        self.config = site_config
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.current_base_url: str | None = None
        self.logger = logging.getLogger(__name__)

        self.url_mappings: dict[str, dict[str, str]] = {}
        self._load_url_mappings()

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info("Initialized Parser for %s", self.site)

    @staticmethod
    def setup_logging() -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )

    def _load_url_mappings(self) -> None:
        mapping_file = Path(self.input_dir) / "url_mappings.json"
        if mapping_file.exists():
            try:
                with mapping_file.open(encoding="utf-8") as f:
                    self.url_mappings = json.load(f)
                self.logger.info("Loaded %d URL mappings", len(self.url_mappings))
            except Exception:
                self.logger.exception("Error loading URL mappings")
        else:
            self.logger.debug("URL mapping file not found: %s", mapping_file)

    def _get_original_url(self, file_path: str | Path) -> str | None:
        file_path_str = str(file_path)
        url = (
            self._try_exact_match(file_path_str)
            or self._try_relative_path_match(file_path_str)
            or self._try_filename_match(file_path_str)
        )
        if not url:
            self.logger.debug("No URL mapping found for %s", file_path)
        return url

    def _try_exact_match(self, file_path_str: str) -> str | None:
        for key, value in self.url_mappings.items():
            if file_path_str.endswith(key):
                return value.get("url")
        return None

    def _try_relative_path_match(self, file_path_str: str) -> str | None:
        try:
            rel_path = os.path.relpath(file_path_str, self.input_dir)
            for path_variant in [rel_path, rel_path.replace("\\", "/")]:
                if path_variant in self.url_mappings:
                    return self.url_mappings[path_variant].get("url")
        except Exception:
            self.logger.debug("Error in relative path matching", exc_info=True)
        return None

    def _try_filename_match(self, file_path_str: str) -> str | None:
        try:
            filename = Path(file_path_str).name
            for key, value in self.url_mappings.items():
                if key.endswith(filename):
                    return value.get("url")
        except Exception:
            self.logger.debug("Error in filename matching", exc_info=True)
        return None

    def _build_tree(self, html_content: str) -> Any:
        parser = html.HTMLParser()
        if html_content.startswith("<?xml"):
            return html.fromstring(html_content.encode("utf-8"), parser=parser)
        return html.fromstring(html_content, parser=parser)

    def _parse_html(self, html_content: str) -> tuple[str, str]:
        title = "Untitled"
        markdown_content = ""
        try:
            tree: Any = self._build_tree(html_content)
            parser_config = self.config.parser

            # Strip unwanted elements
            for selector in parser_config.strip_selectors:
                for element in tree.cssselect(selector):
                    parent = element.getparent()
                    if parent is not None:
                        parent.remove(element)

            # Extract title
            title_result = tree.xpath(parser_config.title_selector)
            title_element = title_result[0] if title_result else None
            if title_element is not None:
                title_text = title_element.text_content()
                title = title_text.strip() if title_text else "Untitled"

            # Find content section
            content_elements = list(tree.cssselect(parser_config.content_selector))
            content_section = content_elements[0] if content_elements else None

            if content_section is not None:
                content_html = html.tostring(content_section, encoding="unicode", pretty_print=True)
                self.logger.info("Successfully extracted content section")
            elif parser_config.fallback_to_body:
                body_result = tree.xpath("//body")
                content_html = (
                    html.tostring(body_result[0], encoding="unicode", pretty_print=True)
                    if body_result
                    else html_content
                )
                self.logger.warning("Could not find specific content section, using body content")
            else:
                self.logger.warning("No content found and no fallback configured")
                content_html = ""

            if content_html:
                content_html = self._convert_relative_links_to_absolute(content_html)
                markdown_content = self._html_to_markdown(content_html)
            else:
                markdown_content = ""

        except Exception:
            self.logger.exception("Error parsing HTML")
            return "Error Parsing Page", "Error parsing the HTML content"

        return title, markdown_content

    @staticmethod
    def _convert_element_link_to_absolute(
        element: Any,
        attribute: str,
        base_url: str,
        absolute_prefixes: tuple[str, ...],
    ) -> bool:
        link = element.get(attribute)
        if not link or link.startswith(absolute_prefixes):
            return False
        absolute_url = urljoin(base_url, link)
        element.set(attribute, absolute_url)
        return True

    def _convert_relative_links_to_absolute(self, html_content: str) -> str:
        if not self.current_base_url:
            return html_content
        try:
            tree: Any = html.fromstring(html_content)
            href_prefixes = ("http://", "https://", "//", "mailto:", "#", "tel:")
            src_prefixes = ("http://", "https://", "//", "data:")
            for element in tree.iter(tag="*"):
                href = element.get("href")
                if href is not None:
                    self._convert_element_link_to_absolute(element, "href", self.current_base_url, href_prefixes)
                src = element.get("src")
                if src is not None:
                    self._convert_element_link_to_absolute(element, "src", self.current_base_url, src_prefixes)
            result = html.tostring(tree, encoding="unicode", pretty_print=True)
            return str(result)
        except Exception:
            self.logger.exception("Error converting relative links")
            return html_content

    def _html_to_markdown(self, html_content: str) -> str:
        config = self.config.parser.markdown
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = config.ignore_links
        text_maker.body_width = config.body_width
        text_maker.protect_links = config.protect_links
        text_maker.unicode_snob = config.unicode_snob
        text_maker.ignore_images = config.ignore_images
        text_maker.ignore_tables = config.ignore_tables
        return text_maker.handle(html_content)

    def _construct_base_url_from_path(self, _file_path: str) -> str:
        return str(self.config.base_url)

    def _extract_domain_from_path(self, file_path: str | Path) -> str:
        try:
            rel_path = Path(file_path).relative_to(Path(self.input_dir))
            path_parts = rel_path.parts
            return path_parts[0] if path_parts else "unknown"
        except ValueError:
            return "unknown"

    def _create_directory_scope(self) -> DirectoryScope:
        return DirectoryScope(self.input_dir, self.input_dir)

    def _parse_file_with_context(self, html_file: str | Path) -> dict[str, Any] | None:
        try:
            return self.parse_file(html_file, preserve_url_context=True)
        except Exception:
            self.logger.exception("Error parsing %s with context", html_file)
            return None

    def parse_file(
        self,
        html_file: str | Path,
        preserve_url_context: bool = False,
    ) -> dict[str, Any] | None:
        original_url = self._get_original_url(html_file)
        original_base_url = self.current_base_url if not preserve_url_context else None
        self.current_base_url = original_url
        if not self.current_base_url:
            self.current_base_url = self._construct_base_url_from_path(str(html_file))

        html_file_path = Path(html_file)
        self.logger.info("Parsing %s", html_file_path)

        try:
            with html_file_path.open(encoding="utf-8") as f:
                html_content = f.read()

            domain = self._extract_domain_from_path(html_file_path)
            output_filename = self._get_output_filename(html_file_path)
            title, content = self._parse_html(html_content)
            metadata = self._create_metadata(html_file_path, title)
            output_path = self._save_markdown(output_filename, title, content, metadata)

            return {
                "source_file": str(html_file_path),
                "output_file": output_path,
                "title": title,
                "domain": domain,
            }
        except Exception:
            self.logger.exception("Error parsing %s", html_file_path)
            return None
        finally:
            if original_base_url is not None and not preserve_url_context:
                self.current_base_url = original_base_url

    def _create_metadata(self, file_path: str | Path, title: str) -> dict[str, Any]:
        domain = self._extract_domain_from_path(file_path)
        metadata: dict[str, Any] = {
            "source_file": str(file_path),
            "title": title,
            "domain": domain,
            "parse_timestamp": datetime.now(UTC).isoformat(),
            "parser": self.__class__.__name__,
        }
        original_url = self._get_original_url(file_path)
        if original_url:
            metadata["source_url"] = original_url
        return metadata

    def _get_output_filename(self, html_file_path: Path) -> str:
        try:
            rel_path = html_file_path.relative_to(Path(self.input_dir))
            parts = rel_path.parts
            if len(parts) >= MIN_NESTED_PATH_PARTS:
                return str(rel_path.with_suffix(""))
        except ValueError:
            pass
        return html_file_path.stem

    def _save_markdown(
        self,
        filename: str,
        title: str,
        content: str,
        metadata: dict[str, Any],
    ) -> str:
        if not filename.endswith(".md"):
            filename = f"{filename}.md"
        output_path = Path(self.output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frontmatter = yaml.dump(metadata, default_flow_style=False)
        markdown_content = f"---\n{frontmatter}---\n\n# {title}\n\n{content}\n"
        with output_path.open("w", encoding="utf-8") as f:
            f.write(markdown_content)
        self.logger.info("Saved markdown to %s", output_path)
        return str(output_path)

    def parse_all(self) -> list[dict[str, Any]]:
        self.logger.info(
            "Parsing HTML files for site '%s' from directory '%s'",
            self.site,
            self.input_dir,
        )
        with self._create_directory_scope() as scoped_dir:
            results: list[dict[str, Any]] = []
            html_files = list(Path(scoped_dir).rglob("*.html"))
            self.logger.info("Found %d HTML files to parse", len(html_files))
            for html_file in html_files:
                try:
                    result = self._parse_file_with_context(html_file)
                    if result:
                        results.append(result)
                except Exception:
                    self.logger.exception("Error parsing %s", html_file)
            if results:
                self._create_index(results)
            self.logger.info("Parsed %d files", len(results))
            return results

    def _create_index(self, results: list[dict[str, Any]]) -> str:
        index_path = Path(self.output_dir) / "index.md"
        with index_path.open("w", encoding="utf-8") as f:
            f.write(f"# {self.site or 'Site'} Parsed Content Index\n\n")
            f.write(f"Total pages parsed: {len(results)}\n\n")
            f.write("| Title | Source File | Output File |\n")
            f.write("|-------|-------------|-------------|\n")
            for result in results:
                title = result.get("title", "Untitled")
                source = Path(result.get("source_file", "")).name
                output = Path(result.get("output_file", "")).name
                f.write(f"| {title} | {source} | [{output}]({output}) |\n")
        self.logger.info("Created index at %s", index_path)
        return str(index_path)

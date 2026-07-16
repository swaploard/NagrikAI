from __future__ import annotations

from pydantic import BaseModel, HttpUrl


class CrawlerConfig(BaseModel):
    max_concurrency: int = 5
    request_delay: float = 1.0
    timeout: int = 30
    max_depth: int = 1
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


class MarkdownConfig(BaseModel):
    ignore_links: bool = False
    body_width: int = 0
    protect_links: bool = True
    unicode_snob: bool = True
    ignore_images: bool = False
    ignore_tables: bool = False


class ParserConfig(BaseModel):
    content_selector: str = "main"
    title_selector: str = "//title"
    strip_selectors: list[str] = ["nav", "footer", "script", "style"]
    fallback_to_body: bool = True
    output_format: str = "markdown"
    markdown: MarkdownConfig = MarkdownConfig()


class SiteConfig(BaseModel):
    name: str
    base_url: HttpUrl
    start_urls: list[HttpUrl]
    allowed_domains: list[str]
    crawler: CrawlerConfig = CrawlerConfig()
    parser: ParserConfig = ParserConfig()


class NagrikAIConfig(BaseModel):
    sites: list[SiteConfig]
    chroma_persist_dir: str = "chroma_db"
    content_dir: str = "content"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

from __future__ import annotations

from pydantic import BaseModel, HttpUrl


class CrawlerConfig(BaseModel):
    max_concurrency: int = 5
    request_delay: float = 1.0
    timeout: int = 30
    max_depth: int = 0  # 0 means unlimited
    user_agent: str = "NagrikAI/0.1 (+https://github.com/nagrik-ai/nagrik-ai)"


class ParserConfig(BaseModel):
    content_selector: str = "main"
    strip_selectors: list[str] = ["nav", "footer", "script", "style"]
    output_format: str = "markdown"


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

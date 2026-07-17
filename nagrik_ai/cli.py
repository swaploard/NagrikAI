from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from nagrik_ai.config.settings import CHROMA_PERSIST_DIR, CONTENT_DIR
from nagrik_ai.factories import create_chroma_store, create_config_manager, create_orchestrator
from nagrik_ai.parser.parser import Parser
from nagrik_ai.utils.text_utils import HybridMarkdownSplitter
from nagrik_ai.vectorstore.vectorizer import MarkdownVectorizer

app = typer.Typer(name="nagrik-ai")
crawl_app = typer.Typer(name="crawl")
app.add_typer(crawl_app, name="crawl", help="Crawl websites for immigration data")
parse_app = typer.Typer(name="parse")
app.add_typer(parse_app, name="parse", help="Parse crawled HTML into Markdown")
vectorize_app = typer.Typer(name="vectorize")
app.add_typer(vectorize_app, name="vectorize", help="Vectorize parsed documents into ChromaDB")


@crawl_app.command()
def sites(
    config_path: Annotated[Path | None, typer.Option("--config", "-c", help="Path to site config YAML")] = None,
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = CONTENT_DIR,
    depth: Annotated[int | None, typer.Option("--depth", "-d", help="Max crawl depth (0 = unlimited)")] = 1,
    manage: Annotated[bool, typer.Option("--manage", "-m", help="Skip already-crawled URLs")] = False,
) -> None:
    from nagrik_ai.crawler.scrapy_runner import run_batch_scrapy_crawl

    from nagrik_ai.config.config_models import SiteConfig

    config_manager = create_config_manager(config_path)
    config = config_manager.load()
    # Each job is a tuple: (site_config, output_directory, manage_flag)
    jobs: list[tuple[SiteConfig, Path, bool]] = []
    for site in config.sites:
        typer.echo(f"Crawling {site.name}...")
        crawled_dir = output_dir / site.name / "crawled"
        jobs.append((site, crawled_dir, manage))
    run_batch_scrapy_crawl(jobs, max_depth=depth)


@parse_app.command()
def all(
    content_dir: Annotated[Path, typer.Option("--content-dir", "-d", help="Content directory")] = CONTENT_DIR,
    config_path: Annotated[Path | None, typer.Option("--config", "-c", help="Path to site config YAML")] = None,
) -> None:
    from nagrik_ai.config.config_models import ParserConfig, SiteConfig

    from pydantic import HttpUrl

    config_manager = create_config_manager(config_path)
    config = config_manager.load()
    site_configs = {site.name: site for site in config.sites}
    default_site = SiteConfig(
        name="default",
        base_url=HttpUrl("https://example.com"),
        start_urls=[],
        allowed_domains=[],
        parser=ParserConfig(),
    )

    for site_dir in sorted(content_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        crawled = site_dir / "crawled"
        parsed = site_dir / "parsed"
        if not crawled.exists():
            continue
        site_name = site_dir.name
        site = site_configs.get(site_name, default_site)
        typer.echo(f"Parsing {site_name}...")
        parser = Parser(
            site_name=site_name,
            site_config=site,
            input_dir=str(crawled),
            output_dir=str(parsed),
        )
        results = parser.parse_all()
        typer.echo(f"  Parsed {len(results)} files")


@vectorize_app.callback(invoke_without_command=True)
def vectorize(
    ctx: typer.Context,
    content_dir: Annotated[Path, typer.Option("--content-dir", "-d", help="Content directory")] = CONTENT_DIR,
    persist_dir: Annotated[
        Path, typer.Option("--persist-dir", "-p", help="ChromaDB persist directory")
    ] = CHROMA_PERSIST_DIR,
    chunk_size: Annotated[int, typer.Option("--chunk-size", "-s", help="Chunk size")] = 512,
    chunk_overlap: Annotated[int, typer.Option("--chunk-overlap", "-o", help="Chunk overlap")] = 64,
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    chroma_store = create_chroma_store(persist_dir)
    text_splitter = HybridMarkdownSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorizer = MarkdownVectorizer(chroma_store, text_splitter=text_splitter)
    for site_dir in sorted(content_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        parsed = site_dir / "parsed"
        if not parsed.exists():
            continue
        typer.echo(f"Vectorizing {site_dir.name}...")
        count = vectorizer.process_directory(str(parsed))
        typer.echo(f"  Added {count} files")


@app.command()
def app_command(
    persist_dir: Annotated[
        Path, typer.Option("--persist-dir", "-p", help="ChromaDB persist directory")
    ] = CHROMA_PERSIST_DIR,
    share: Annotated[bool, typer.Option("--share", help="Create a public link")] = False,
    port: Annotated[int, typer.Option("--port", help="Port to run on")] = 7860,
) -> None:
    from nagrik_ai.app import launch

    orch = create_orchestrator()
    launch(orch, share=share, server_port=port)
    _ = persist_dir  # used by Typer option


if __name__ == "__main__":
    app()

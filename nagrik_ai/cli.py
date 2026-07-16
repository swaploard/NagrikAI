from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from nagrik_ai.config.settings import CHROMA_PERSIST_DIR, CONTENT_DIR
from nagrik_ai.crawler.runner import run_crawl
from nagrik_ai.factories import create_chroma_store, create_config_manager, create_orchestrator
from nagrik_ai.parser.parser import HTMLParser
from nagrik_ai.vectorstore.vectorizer import Vectorizer

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
    depth: Annotated[int | None, typer.Option("--depth", "-d", help="Max crawl depth (0 = unlimited)")] = None,
    manage: Annotated[bool, typer.Option("--manage", "-m", help="Skip already-crawled URLs")] = False,
) -> None:
    config_manager = create_config_manager(config_path)
    config = config_manager.load()
    for site in config.sites:
        typer.echo(f"Crawling {site.name}...")
        count = run_crawl(site, output_dir, max_depth=depth, manage=manage)
        typer.echo(f"  Saved {count} files")


@parse_app.command()
def all(
    content_dir: Annotated[Path, typer.Option("--content-dir", "-d", help="Content directory")] = CONTENT_DIR,
) -> None:
    parser = HTMLParser()
    for site_dir in content_dir.iterdir():
        if not site_dir.is_dir():
            continue
        crawled = site_dir / "crawled"
        parsed = site_dir / "parsed"
        if not crawled.exists():
            continue
        typer.echo(f"Parsing {site_dir.name}...")
        count = 0
        for md_file in sorted(crawled.glob("*.md")):
            doc = parser.parse_file(md_file, site=site_dir.name)
            parser.save_parsed(doc, parsed)
            count += 1
        typer.echo(f"  Parsed {count} files")


@vectorize_app.command()
def run(
    content_dir: Annotated[Path, typer.Option("--content-dir", "-d", help="Content directory")] = CONTENT_DIR,
    persist_dir: Annotated[
        Path, typer.Option("--persist-dir", "-p", help="ChromaDB persist directory")
    ] = CHROMA_PERSIST_DIR,
    chunk_size: Annotated[int, typer.Option("--chunk-size", "-s", help="Chunk size")] = 512,
    chunk_overlap: Annotated[int, typer.Option("--chunk-overlap", "-o", help="Chunk overlap")] = 64,
) -> None:
    chroma_store = create_chroma_store(persist_dir)
    vectorizer = Vectorizer(chroma_store, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for site_dir in sorted(content_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        parsed = site_dir / "parsed"
        if not parsed.exists():
            continue
        typer.echo(f"Vectorizing {site_dir.name}...")
        count = vectorizer.vectorize_directory(parsed, site=site_dir.name)
        typer.echo(f"  Added {count} chunks")


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

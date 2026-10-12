from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

import typer

from nagrik_ai.config.config_models import CHROMA_PERSIST_DIR, CONTENT_DIR
from nagrik_ai.factories import build_app_stack, create_chroma_store, create_config_manager, warmup
from nagrik_ai.parser.parser import Parser
from nagrik_ai.services.tracing import get_tracer
from nagrik_ai.utils.text_utils import HybridMarkdownSplitter
from nagrik_ai.vectorstore.vectorizer import MarkdownVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = typer.Typer(name="nagrik-ai")
crawl_app = typer.Typer(name="crawl")
app.add_typer(crawl_app, name="crawl", help="Crawl websites for taxation data")
parse_app = typer.Typer(name="parse")
app.add_typer(parse_app, name="parse", help="Parse crawled HTML into Markdown")
vectorize_app = typer.Typer(name="vectorize")
app.add_typer(vectorize_app, name="vectorize", help="Vectorize parsed documents into ChromaDB")
acceptance_app = typer.Typer(name="acceptance")
app.add_typer(acceptance_app, name="acceptance", help="Audit a bounded CBIC rollout")


@crawl_app.command()
def sites(
    config_path: Annotated[Path | None, typer.Option("--config", "-c", help="Path to site config YAML")] = None,
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = CONTENT_DIR,
    depth: Annotated[int | None, typer.Option("--depth", "-d", help="Override each site's max crawl depth")] = None,
    page_cap: Annotated[
        int | None,
        typer.Option("--page-cap", min=1, help="Stop after this many downloaded responses"),
    ] = None,
    document_cap: Annotated[
        int | None,
        typer.Option("--document-cap", min=1, help="Stop after this many emitted document items"),
    ] = None,
    site_names: Annotated[
        list[str] | None,
        typer.Option("--site", "-s", help="Site name(s) to crawl (can be repeated, e.g. -s gst -s gst_tutorial)"),
    ] = None,
    manage: Annotated[bool, typer.Option("--manage", "-m", help="Skip already-crawled URLs")] = False,
    force: Annotated[bool, typer.Option("--force", help="Fetch known URLs again")] = False,
    incremental: Annotated[
        bool, typer.Option("--incremental", help="Discover new records and revisit known URLs by policy")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if manage and force:
        typer.echo("--manage and --force cannot be used together.", err=True)
        raise typer.Exit(2)

    from nagrik_ai.config.config_models import SiteConfig
    from nagrik_ai.crawler.scrapy_runner import run_batch_scrapy_crawl

    config_manager = create_config_manager(config_path)
    config = config_manager.load()
    sites_to_crawl = [s for s in config.sites if site_names is None or s.name in site_names]
    if not sites_to_crawl:
        typer.echo("No matching sites found.", err=True)
        raise typer.Exit(1)
    if site_names:
        missing = set(site_names) - {s.name for s in sites_to_crawl}
        if missing:
            typer.echo(f"Unknown site(s): {', '.join(sorted(missing))}", err=True)
            raise typer.Exit(1)
    # Each job is a tuple: (site_config, output_directory, manage_flag)
    jobs: list[tuple[SiteConfig, Path, bool]] = []
    for site in sites_to_crawl:
        if document_cap is not None:
            site = site.model_copy(
                update={
                    "discovery": site.discovery.model_copy(
                        update={"max_documents": min(site.discovery.max_documents, document_cap)}
                    )
                }
            )
        typer.echo(f"Crawling {site.name}...")
        crawled_dir = output_dir / site.name / "crawled"
        jobs.append((site, crawled_dir, manage))
    run_batch_scrapy_crawl(
        jobs,
        max_depth=depth,
        force=force,
        incremental=incremental,
        page_limit=page_cap,
        document_limit=document_cap,
    )


@crawl_app.command("reconcile")
def reconcile_manifest(
    site_root: Annotated[Path, typer.Argument(help="Site content directory containing crawled/ and manifests/")],
) -> None:
    from nagrik_ai.crawler.crawl_manifest import CrawlManifest

    manifest = CrawlManifest(site_root)
    reconciled = manifest.reconcile_migrations()
    typer.echo(f"Reconciled {len(reconciled)} migration(s).")


@parse_app.command()
def all(
    content_dir: Annotated[Path, typer.Option("--content-dir", "-d", help="Content directory")] = CONTENT_DIR,
    config_path: Annotated[Path | None, typer.Option("--config", "-c", help="Path to site config YAML")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    from pydantic import HttpUrl

    from nagrik_ai.config.config_models import ParserConfig, SiteConfig

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
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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
    from nagrik_ai.vectorstore.index_snapshot import capture_index_snapshot, write_index_snapshot

    snapshot_path = persist_dir / "acceptance_index_snapshot.json"
    snapshot = capture_index_snapshot(chroma_store)
    write_index_snapshot(snapshot, snapshot_path)
    typer.echo(f"Index snapshot: {snapshot_path} ({snapshot.total_entries} entries)")


@acceptance_app.command("audit")
def acceptance_audit(
    site_root: Annotated[Path, typer.Argument(help="Site directory containing crawled/, parsed/, and manifests/")],
    page_cap: Annotated[int, typer.Option("--page-cap", min=1, max=50)] = 20,
    document_cap: Annotated[int, typer.Option("--document-cap", min=1, max=100)] = 25,
    depth: Annotated[int, typer.Option("--depth", min=0, max=2)] = 0,
    sample_size: Annotated[int, typer.Option("--sample-size", min=1, max=25)] = 5,
    review_ledger: Annotated[Path | None, typer.Option("--review-ledger")] = None,
    index_db: Annotated[Path | None, typer.Option("--index-db")] = None,
    vector_snapshot: Annotated[Path | None, typer.Option("--vector-snapshot")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    gate: Annotated[bool, typer.Option("--gate", help="Exit unsuccessfully unless every rollout gate passes")] = False,
) -> None:
    from nagrik_ai.acceptance import CrawlLimits, build_acceptance_report, write_acceptance_report

    report_path = output or site_root / "reports" / "acceptance.json"
    report = build_acceptance_report(
        site_root,
        CrawlLimits(page_cap=page_cap, document_cap=document_cap, depth=depth),
        sample_size=sample_size,
        review_ledger_path=review_ledger,
        index_db_path=index_db,
        vector_snapshot_path=vector_snapshot,
    )
    write_acceptance_report(report, report_path)
    typer.echo(f"Acceptance report: {report_path}")
    typer.echo(
        "Manifest outcomes: "
        + ", ".join(f"{status}={count}" for status, count in report.manifest_status_counts.items())
    )
    typer.echo(
        f"Corpus: sources={report.snapshot.source_count}, versions={report.snapshot.content_version_count}, "
        f"normalized={report.snapshot.normalized_document_count}, chunks={report.snapshot.chunk_count}"
    )
    typer.echo(f"Rollout ready: {'yes' if report.rollout_ready else 'no'}")
    if gate and not report.rollout_ready:
        for error in report.validation_errors:
            typer.echo(f"- {error}", err=True)
        raise typer.Exit(1)


@acceptance_app.command("compare")
def acceptance_compare(
    first: Annotated[Path, typer.Argument(help="Acceptance report from the first full run")],
    second: Annotated[Path, typer.Argument(help="Acceptance report from the second full run")],
) -> None:
    from nagrik_ai.acceptance import compare_acceptance_reports

    differences = compare_acceptance_reports(first, second)
    if differences:
        typer.echo(f"Non-idempotent fields: {', '.join(differences)}", err=True)
        raise typer.Exit(1)
    typer.echo("Semantic corpus and index state are idempotent.")


trace_app = typer.Typer(name="trace")
app.add_typer(trace_app, name="trace", help="Test LangSmith tracing")


@trace_app.command()
def test(
    query: Annotated[str, typer.Argument(help="Test query for tracing")] = "What is QRMP?",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from nagrik_ai.agent.rag_graph import run_rag_query

    tracer = get_tracer()
    if not tracer.enabled:
        typer.echo(
            "LangSmith tracing is not enabled. Set LANGSMITH_TRACING_ENABLED=true "
            "and LANGSMITH_API_KEY in your environment.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"LangSmith tracing enabled (project: {tracer._project_name})")
    typer.echo(f"Running test query: {query!r}")
    result = run_rag_query(query, tracer=tracer)
    typer.echo(f"Response ({len(result.response)} chars): {result.response[:200]}...")
    typer.echo(f"Sources: {len(result.sources)}")
    typer.echo(f"Latency: {result.latency_ms:.1f}ms")
    typer.echo("Check LangSmith dashboard for traces.")


@app.command()
def app_command(
    persist_dir: Annotated[
        Path, typer.Option("--persist-dir", "-p", help="ChromaDB persist directory")
    ] = CHROMA_PERSIST_DIR,
    share: Annotated[bool, typer.Option("--share", help="Create a public link")] = False,
    port: Annotated[int, typer.Option("--port", help="Port to run on")] = 7860,
    llm_provider: Annotated[
        str | None, typer.Option("--llm-provider", help="LLM provider: ollama or openrouter")
    ] = None,
) -> None:
    from nagrik_ai.app import launch

    if llm_provider is None:
        config_manager = create_config_manager()
        config = config_manager.load()
        llm_provider = config.llm_provider

    typer.echo("🚀 Starting Gradio app")
    typer.echo(f"📚 Using ChromaDB collection 'nagrik_ai_docs' from '{persist_dir}'")
    typer.echo(f"🤖 LLM Provider: {llm_provider}")

    typer.echo("⏳ Building app stack (loading models, wiring dependencies)...")
    stack_start = time.perf_counter()
    stack = build_app_stack(persist_dir=persist_dir, enable_fallback=True)
    typer.echo(f"✅ App stack built in {time.perf_counter() - stack_start:.1f}s")

    typer.echo("🔥 Warming up (embed query, BM25 index, reranker first predict)...")
    warmup_start = time.perf_counter()
    warmup(stack)
    typer.echo(f"✅ Warmup complete in {time.perf_counter() - warmup_start:.1f}s")

    launch(stack=stack, share=share, server_port=port)


agent_app = typer.Typer(name="agent")
app.add_typer(agent_app, name="agent", help="Chat with the AI agent using tool calling")


@agent_app.command()
def chat(
    query: str = typer.Argument(help="Your question"),
    session: Annotated[str | None, typer.Option("--session", "-s", help="Session / thread ID to resume")] = None,
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="Interactive multi-turn mode")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from nagrik_ai.agent.router import run_agent

    if interactive:
        _interactive_chat(session_id=session, verbose=verbose)
        return

    answer, tid = run_agent(query, thread_id=session)
    typer.echo(f"[session: {tid}]")
    typer.echo(answer)


def _interactive_chat(session_id: str | None = None, verbose: bool = False) -> None:
    from nagrik_ai.agent.router import run_agent

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tid: str | None = session_id
    typer.echo("Interactive chat mode. Type /quit or press Ctrl+C to exit.")

    try:
        while True:
            query = typer.prompt("You")
            if query.strip().lower() in ("/quit", "/exit", "/q"):
                break
            if not query.strip():
                continue

            answer, tid = run_agent(query, thread_id=tid)
            typer.echo(f"[session: {tid}]")
            typer.echo(f"Assistant: {answer}")
            typer.echo()
    except (EOFError, KeyboardInterrupt):
        pass

    if tid:
        typer.echo(f"\nSession ID: {tid}  (use --session {tid} to resume)")


if __name__ == "__main__":
    app()

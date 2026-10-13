"""Run the production parse/index/retrieval workflow twice on a retained smoke corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nagrik_ai.acceptance import (
    CrawlLimits,
    build_acceptance_report,
    compare_acceptance_reports,
    write_acceptance_report,
)
from nagrik_ai.config.config_manager import ConfigManager
from nagrik_ai.config.config_models import EMBEDDING_MODEL
from nagrik_ai.factories import create_chroma_store
from nagrik_ai.parser.parser import Parser
from nagrik_ai.services.document_retrieval_service import DocumentRetrievalService
from nagrik_ai.services.tracing import LangSmithTracer
from nagrik_ai.utils.text_utils import HybridMarkdownSplitter
from nagrik_ai.vectorstore.bm25_retriever import BM25Retriever
from nagrik_ai.vectorstore.index_outbox import IndexOutbox
from nagrik_ai.vectorstore.index_snapshot import capture_index_snapshot, write_index_snapshot
from nagrik_ai.vectorstore.vectorizer import MarkdownVectorizer


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("site_root", type=Path)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--persist-dir", type=Path, required=True)
    args = cli.parse_args()
    site_root: Path = args.site_root
    config = ConfigManager(args.config).load()
    site = next(site for site in config.sites if site.name == site_root.name)
    store = create_chroma_store(args.persist_dir)
    outbox = IndexOutbox(args.persist_dir / "index_state.sqlite")
    vectorizer = MarkdownVectorizer(
        store, HybridMarkdownSplitter(), index_outbox=outbox,
        embedding_provider="huggingface", embedding_model=EMBEDDING_MODEL, embedding_model_version=EMBEDDING_MODEL,
    )
    reports: list[Path] = []
    for run_number in (1, 2):
        Parser(site.name, site, str(site_root / "crawled"), str(site_root / "parsed")).parse_all()
        vectorizer.process_directory(str(site_root / "parsed"))
        vectorizer.index_reconciler().reconcile()
        snapshot_path = args.persist_dir / "acceptance_index_snapshot.json"
        write_index_snapshot(capture_index_snapshot(store), snapshot_path)
        retriever = DocumentRetrievalService(
            store, hybrid_search=True, bm25_retriever=BM25Retriever(store), tracer=LangSmithTracer(enabled=False),
        )
        report = build_acceptance_report(
            site_root, CrawlLimits(page_cap=3, document_cap=3),
            index_db_path=outbox.path, vector_snapshot_path=snapshot_path, retrieval_service=retriever,
        )
        report_path = site_root / "reports" / f"validation-{run_number}.json"
        write_acceptance_report(report, report_path)
        reports.append(report_path)
        print(json.dumps({
            "run": run_number, "snapshot": report.snapshot.model_dump(),
            "probes": [probe.model_dump() for probe in report.retrieval_probes],
            "validation_errors": report.validation_errors,
        }, indent=2))
    differences = compare_acceptance_reports(*reports)
    print(json.dumps({"idempotency_differences": differences, "pending_operations": len(outbox.pending())}))
    if differences or outbox.pending():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

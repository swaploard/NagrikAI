"""Example 3: Direct Tool Invocation — Real Tools, No LLM Router.

Unlike `01_basic_routing.py` and `02_web_search.py` (which only exercise `run_agent`
and let the LLM decide which tool to call), this script invokes each real tool function
DIRECTLY with fixed arguments. No LLM routing is involved by default — you see exactly
what each tool returns.

Tools demonstrated (one demo each):
  - rag_search   -> nagrik_ai.tools.rag_tool.rag_search_with_sources (real ChromaDB)
  - web_search   -> nagrik_ai.tools.web_search.web_search            (real Tavily API)
  - read_pdf     -> nagrik_ai.tools.pdf_reader.read_pdf              (pypdf)

Usage:
  uv run python example/03_direct_tools.py --all
  uv run python example/03_direct_tools.py --rag
  uv run python example/03_direct_tools.py --web
  uv run python example/03_direct_tools.py --pdf
  uv run python example/03_direct_tools.py --pdf --pdf-path /path/to/file.pdf
  uv run python example/03_direct_tools.py --all --synthesize   # optional LLM step

Requirements:
  - rag_search: vectorized ChromaDB (already present in this workspace).
  - web_search: NAGRIKAI_TAVILY_API_KEY (already set in .env for this workspace).
  - read_pdf:   none (a minimal sample PDF is generated inline if --pdf-path is omitted).
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Minimal valid one-page PDF (no external dependency needed to author it).
_SAMPLE_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Resources"
    b"<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
    b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"5 0 obj<</Length 88>>stream\n"
    b"BT /F1 18 Tf 72 700 Td (NagrikAI sample PDF for read_pdf demo.) Tj "
    b"0 -28 Td (GST is a destination-based indirect tax.) Tj ET\n"
    b"endstream endobj\n"
    b"xref\n"
    b"0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000052 00000 n \n"
    b"0000000101 00000 n \n"
    b"0000000196 00000 n \n"
    b"0000000243 00000 n \n"
    b"trailer<</Size 6/Root 1 0 R>>\n"
    b"startxref\n"
    b"357\n"
    b"%%EOF\n"
)


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def demo_rag_search() -> dict[str, object]:
    """Call rag_search_with_sources directly against the real ChromaDB."""
    from nagrik_ai.tools.rag_tool import rag_search_with_sources

    _section("rag_search (real ChromaDB)")
    query = "What is GST input tax credit?"
    print(f"Query: {query!r}\n")

    result = rag_search_with_sources(query)

    print("RESPONSE:")
    print(result["response"])
    print(f"\nLATENCY_MS: {result['latency_ms']}  "
          f"TOTAL_CHUNKS: {result['total_chunks_retrieved']}  "
          f"CITATIONS_VALID: {result['citations_valid']}")
    print("\nSOURCES:")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src['title']}  ({src['domain']})  score={src['score']:.4f}")
        print(f"      {src['url']}")

    if not result["sources"]:
        print("\nNOTE: No sources returned. The vector store may be empty — "
              "run `uv run nagrik-ai vectorize run` first.")

    return result


def demo_web_search() -> str:
    """Call web_search directly against the real Tavily API."""
    from nagrik_ai.tools.web_search import web_search

    _section("web_search (real Tavily API)")
    api_key = os.getenv("NAGRIKAI_TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("SKIPPED: NAGRIKAI_TAVILY_API_KEY / TAVILY_API_KEY is not set.")
        print("Set it in your .env to run this demo against the live Tavily API.")
        return ""

    query = "FAQs on Mandatory Capture of Ship-to Field and Voluntary Closure of E-Way Bill, 2026"
    print(f"Query: {query!r}\n")

    result = web_search(query)

    print("RESULT:")
    print(result)
    return result


def demo_read_pdf(pdf_path: str | None = None) -> str:
    """Call read_pdf directly on a sample PDF (generated inline if no path given)."""
    from nagrik_ai.tools.pdf_reader import read_pdf

    _section("read_pdf (pypdf)")

    if pdf_path:
        target = pdf_path
        print(f"Reading provided PDF: {target}\n")
    else:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(_SAMPLE_PDF)
            target = tmp.name
        print(f"Reading generated sample PDF: {target}\n")

    result = read_pdf(target)

    preview = result[:500]
    print("EXTRACTED TEXT (first 500 chars):")
    print(preview)
    print(f"\nTOTAL CHARS EXTRACTED: {len(result)}")

    if not pdf_path:
        Path(target).unlink(missing_ok=True)

    return result


def synthesize(results: dict[str, object]) -> None:
    """Optional LLM step: combine tool outputs into one answer (off by default)."""
    from nagrik_ai.services.llm_service import create_llm_service

    _section("synthesize (optional LLM step)")
    rag = results.get("rag")
    web = results.get("web")
    pdf = results.get("pdf")

    parts: list[str] = []
    if isinstance(rag, dict):
        parts.append(f"[rag_search]\n{rag.get('response', '')}")
    if isinstance(web, str) and web:
        parts.append(f"[web_search]\n{web}")
    if isinstance(pdf, str) and pdf:
        parts.append(f"[read_pdf]\n{pdf[:1000]}")

    if not parts:
        print("No tool results to synthesize.")
        return

    prompt = (
        "You are given the raw outputs of several tools. Write a single, concise, "
        "well-structured answer that integrates them. Do not invent facts.\n\n"
        + "\n\n".join(parts)
    )

    llm = create_llm_service()
    answer = llm.generate(prompt, system="You are a helpful Indian GST assistant.")
    print("SYNTHESIZED ANSWER:")
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Directly invoke NagrikAI real tools (no LLM router)."
    )
    parser.add_argument("--rag", action="store_true", help="Run rag_search demo")
    parser.add_argument("--web", action="store_true", help="Run web_search demo")
    parser.add_argument("--pdf", action="store_true", help="Run read_pdf demo")
    parser.add_argument("--all", action="store_true", help="Run all three demos")
    parser.add_argument("--synthesize", action="store_true",
                        help="Optionally combine results with an LLM")
    parser.add_argument("--pdf-path", default=None,
                        help="Path to an existing PDF for the read_pdf demo")
    args = parser.parse_args()

    if not (args.rag or args.web or args.pdf or args.all):
        parser.print_help()
        return

    results: dict[str, object] = {}

    if args.rag or args.all:
        results["rag"] = demo_rag_search()
    if args.web or args.all:
        results["web"] = demo_web_search()
    if args.pdf or args.all:
        results["pdf"] = demo_read_pdf(args.pdf_path)

    if args.synthesize:
        synthesize(results)


if __name__ == "__main__":
    main()

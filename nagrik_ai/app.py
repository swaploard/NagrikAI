from __future__ import annotations

import html
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr

from nagrik_ai.factories import AppStack, get_app_stack
from nagrik_ai.models.rag_result import SourceInfo
from nagrik_ai.services.citation_service import make_citations_clickable
from nagrik_ai.services.llm_service import RateLimitError
from nagrik_ai.tools.pdf_reader import read_pdf

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def _format_sources(sources: list[SourceInfo]) -> str:
    """Format sources for display in sources panel."""
    if not sources:
        return "No sources yet..."

    lines: list[str] = []
    for s in sources:
        lines.append(f"**[{s.citation_id}]** [{s.title}]({s.url}) — *{s.domain}* (source: {s.source_id})")
    return "\n\n".join(lines)


def _agent_sources(citations: list[dict[str, Any]]) -> list[SourceInfo]:
    return [
        SourceInfo(
            title=str(c.get("title", c.get("source_id", "Source"))),
            url=str(c.get("url", "")),
            domain=str(c.get("domain", "")),
            source_id=str(c["source_id"]),
            citation_id=int(c["citation_id"]),
            chunk_index=int(c.get("chunk_index", 0)),
            total_chunks=int(c.get("total_chunks", 1)),
        )
        for c in citations
        if c.get("source_id") and c.get("citation_id")
    ]


def _format_claim_verdicts(verdicts: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for verdict in verdicts:
        references = [f"[{cid}]" for cid in verdict.get("evidence_ids", [])]
        references.extend(verdict.get("tool_result_refs", []))
        lines.append(
            f"**{html.escape(verdict['status'])}** ({html.escape(verdict['method'])}): "
            f"{html.escape(verdict['claim'])} — {html.escape(', '.join(references))}"
        )
    return "\n\n".join(lines) or "No claim verdicts available."


def handle_pdf_upload(file: Any) -> tuple[str, str]:
    """Extract text from an uploaded PDF file.

    Args:
        file: A gradio File object (has a ``name`` attribute pointing to the temp path).

    Returns:
        A tuple of (extracted_text, status_message).
    """
    if file is None:
        return "", "No PDF uploaded."
    try:
        result = read_pdf(file.name)
        if not result.ok:
            return "", f"Error reading PDF: {result.error_message}"
        if not isinstance(result.data, dict):
            return "", "Error reading PDF: invalid document result."
        text = str(result.data["document"]["text"])
        filename = Path(file.name).name
        status = f"Uploaded **{filename}** — {len(text)} characters extracted."
        return text, status
    except Exception as e:
        return "", f"Error reading PDF: {e}"


def _build_ui(stack: AppStack | None = None) -> gr.Blocks:
    """Build the Gradio interface used by the launcher."""
    return _build_streaming_ui(stack)


def _build_streaming_ui(stack: AppStack | None = None) -> gr.Blocks:
    """Build a Gradio interface with streaming response support and sources panel."""

    def respond_stream(
        message: str,
        chat_history: list[dict[str, str]],
        session_state: str,
        pdf_context: str,
    ) -> Generator[tuple[str, list[dict[str, str]], str, str, str, str]]:
        history = [*(chat_history or []), {"role": "user", "content": message}]
        sources, steps_display, verdicts, banner = "No sources yet...", "", "", ""
        response = "Working…"
        yield "", [*history, {"role": "assistant", "content": response}], sources, steps_display, verdicts, banner
        try:
            enhanced_query = (
                f"[Uploaded PDF content]\n{pdf_context[:20000]}\n\n[User question]\n{message}"
                if pdf_context
                else message
            )
            stream = (stack or get_app_stack(enable_fallback=True)).stream_agent(
                enhanced_query, thread_id=session_state
            )
            for event in stream:
                if event["type"] == "agent_step":
                    step = event["data"]
                    result = step["tool_result"]
                    authority = f" [{result.source_authority}]" if result and result.source_authority else ""
                    line = f"{step['observation_summary']} `{step['action']}`{authority}"
                    steps_display += ("\n\n" if steps_display else "") + line
                    response = step["observation_summary"]
                elif event["type"] == "final":
                    data = event["data"]
                    source_list = _agent_sources(data.get("citations", []))
                    sources = _format_sources(source_list)
                    response = make_citations_clickable(data["answer"], source_list)
                    verdicts = _format_claim_verdicts(data.get("claim_verdicts", []))
                    if data.get("finalized_with_limitations"):
                        banner = (
                            "> ⚠️ This answer could not be fully validated. Review the claim checks and sources below."
                        )
                yield (
                    "",
                    [*history, {"role": "assistant", "content": response}],
                    sources,
                    steps_display,
                    verdicts,
                    banner,
                )
        except RateLimitError:
            response = "The AI service is temporarily at capacity. Please wait a few moments and try again."
            yield "", [*history, {"role": "assistant", "content": response}], sources, steps_display, verdicts, banner
        except Exception:
            logger.exception("Error generating agent response")
            response = "I encountered an error while processing your query. Please try again."
            yield "", [*history, {"role": "assistant", "content": response}], sources, steps_display, verdicts, banner

    with gr.Blocks(title="NagrikAI - Indian TAX Assistant") as demo:
        session_state = gr.State(value=lambda: str(uuid4()))
        pdf_context = gr.State(value="")

        gr.Markdown("# NagrikAI")
        gr.Markdown("AI-powered Indian taxation assistant. Ask me about tax laws, regulations, and compliance.")

        with gr.Row():
            with gr.Column(scale=3):
                limitations_banner = gr.Markdown("")
                chatbot = gr.Chatbot(label="Conversation", height=500)
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Ask about Indian taxation...",
                    lines=2,
                )

                gr.HTML(
                    """<p style="font-size: 0.8em; color: #666; margin-top: 0.5em;">
                        Information may contain errors. Always verify with official sources.
                    </p>"""
                )

                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear")

                with gr.Accordion("Upload PDF", open=False):
                    pdf_upload = gr.File(label="Choose a PDF file", file_types=[".pdf"])
                    pdf_status = gr.Markdown("No PDF uploaded.")

            with gr.Column(scale=1):
                sources_panel = gr.Markdown(
                    "### Sources\n\nNo sources yet...",
                    elem_id="sources-panel",
                )

                with gr.Accordion("AgentSteps", open=False):
                    steps_panel = gr.Markdown("")
                with gr.Accordion("Claim checks", open=False):
                    claims_panel = gr.Markdown("")

        msg.submit(
            respond_stream,
            [msg, chatbot, session_state, pdf_context],
            [msg, chatbot, sources_panel, steps_panel, claims_panel, limitations_banner],
        )
        submit.click(
            respond_stream,
            [msg, chatbot, session_state, pdf_context],
            [msg, chatbot, sources_panel, steps_panel, claims_panel, limitations_banner],
        )
        clear.click(
            lambda: (
                [],
                "",
                "### Sources\n\nNo sources yet...",
                str(uuid4()),
                "",
                "No PDF uploaded.",
                "",
                "",
                "",
            ),
            None,
            [
                chatbot,
                msg,
                sources_panel,
                session_state,
                pdf_context,
                pdf_status,
                steps_panel,
                claims_panel,
                limitations_banner,
            ],
        )

        pdf_upload.change(handle_pdf_upload, [pdf_upload], [pdf_context, pdf_status])

        gr.Examples(
            examples=[
                "Who all are eligible for the QRMP scheme?",
                "What is IFF?",
                "Whether it is required to exercise the option every quarter / year?",
                "When can Form GSTR-1 be filed as Nil?",
            ],
            inputs=msg,
        )

    return demo  # type: ignore[no-any-return]


def launch(stack: AppStack | None = None, **kwargs: Any) -> None:
    demo = _build_ui(stack)
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch()

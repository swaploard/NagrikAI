from __future__ import annotations

import logging
from collections.abc import Generator, Iterator
from typing import Any

import gradio as gr

from nagrik_ai.factories import create_orchestrator
from nagrik_ai.services.rag_orchestrator import RAGOrchestrator, RAGResponse

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def _format_citations(citations: dict[int, dict[str, Any]]) -> str:
    """Format citations for display in sources panel."""
    if not citations:
        return "No sources yet..."

    lines: list[str] = []
    for num, doc in sorted(citations.items()):
        meta = doc.get("metadata", {})
        title = meta.get("title", "Unknown")
        url = meta.get("citation_url", meta.get("url", "#"))
        domain = meta.get("domain", "unknown")
        source_id = meta.get("source_id", "unknown")
        lines.append(f"**[{num}]** [{title}]({url}) — *{domain}* (source: {source_id})")
    return "\n\n".join(lines)


def _build_ui(orch: RAGOrchestrator) -> gr.Blocks:
    """Build the Gradio interface used by the launcher."""
    return _build_streaming_ui(orch)


def _build_streaming_ui(orch: RAGOrchestrator) -> gr.Blocks:
    """Build a Gradio interface with streaming response support and sources panel."""

    def respond_stream(
        message: str,
        chat_history: list[dict[str, str]],
    ) -> Generator[tuple[str, list[dict[str, str]], str]]:
        if not chat_history:
            chat_history = []

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "..."})
        yield "", chat_history, "No sources yet..."

        citations_display = ""
        try:
            stream: Iterator[RAGResponse] = orch.query_stream(message)
            chat_history.pop()
            assistant_response = ""
            for rag_response in stream:
                assistant_response += rag_response.answer
                if rag_response.citations:
                    citations_display = _format_citations(rag_response.citations)
                current_history = chat_history.copy()
                current_history.append({"role": "assistant", "content": assistant_response})
                yield "", current_history, citations_display

            chat_history.append({"role": "assistant", "content": assistant_response})
            yield "", chat_history, citations_display

        except Exception:
            logger.exception("Error generating response")
            chat_history.pop()
            error_message = "I encountered an error while processing your query. Please try again."
            chat_history.append({"role": "assistant", "content": error_message})
            yield "", chat_history, citations_display

    with gr.Blocks(title="NagrikAI - Indian Immigration Assistant") as demo:
        gr.Markdown("# NagrikAI")
        gr.Markdown(
            "AI-powered Indian immigration assistant. "
            "Ask me about residence permits, social security, taxation, and more."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=500)
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Ask about Indian immigration processes...",
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

            with gr.Column(scale=1):
                sources_panel = gr.Markdown(
                    "### Sources\n\nNo sources yet...",
                    elem_id="sources-panel",
                )

        msg.submit(respond_stream, [msg, chatbot], [msg, chatbot, sources_panel])
        submit.click(respond_stream, [msg, chatbot], [msg, chatbot, sources_panel])
        clear.click(lambda: ([], "", "### Sources\n\nNo sources yet..."), None, [chatbot, msg, sources_panel])

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


def launch(orch: RAGOrchestrator | None = None, **kwargs: Any) -> None:
    orch = orch or create_orchestrator()
    demo = _build_ui(orch)
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch()

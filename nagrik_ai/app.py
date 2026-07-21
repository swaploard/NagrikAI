from __future__ import annotations

import logging
from collections.abc import Generator, Iterator
from typing import Any
from uuid import uuid4

import gradio as gr

from nagrik_ai.factories import create_orchestrator
from nagrik_ai.models.rag_result import RAGResult, SourceInfo
from nagrik_ai.services.citation_service import make_citations_clickable
from nagrik_ai.services.llm_service import RateLimitError
from nagrik_ai.services.rag_orchestrator import RAGOrchestrator

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def _format_sources(sources: list[SourceInfo]) -> str:
    """Format sources for display in sources panel."""
    if not sources:
        return "No sources yet..."

    lines: list[str] = []
    for s in sources:
        lines.append(
            f"**[{s.citation_id}]** [{s.title}]({s.url}) — *{s.domain}* (source: {s.source_id})"
        )
    return "\n\n".join(lines)


def _build_ui(orch: RAGOrchestrator) -> gr.Blocks:
    """Build the Gradio interface used by the launcher."""
    return _build_streaming_ui(orch)


def _build_streaming_ui(orch: RAGOrchestrator) -> gr.Blocks:
    """Build a Gradio interface with streaming response support and sources panel."""

    def respond_stream(
        message: str,
        chat_history: list[dict[str, str]],
        session_state: str,
    ) -> Generator[tuple[str, list[dict[str, str]], str]]:
        if not chat_history:
            chat_history = []

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "..."})
        yield "", chat_history, "No sources yet..."

        try:
            logger.info("Starting to consume response stream")
            stream: Iterator[dict[str, Any]] = orch.query_stream(message, session_id=session_state)
            chat_history.pop()
            assistant_response = ""
            citations_display = ""
            first_token = True
            for chunk in stream:
                if chunk["type"] == "token":
                    if first_token:
                        logger.info("Replacing ellipsis with first meaningful chunk")
                        first_token = False
                    assistant_response += chunk["content"]
                    current_history = chat_history.copy()
                    current_history.append({"role": "assistant", "content": assistant_response})
                    yield "", current_history, citations_display
                elif chunk["type"] == "final":
                    result: RAGResult = chunk["data"]

                    # Phase 2: web search fallback if RAG returned no answer
                    if "I could not find this information" in result.response:
                        logger.info("RAG returned no answer; performing web search fallback")
                        yield "", chat_history, "Searching the web for more information..."
                        try:
                            from nagrik_ai.services.llm_service import create_llm_service
                            from nagrik_ai.tools.web_search import web_search

                            web_result = web_search(message)
                            llm = create_llm_service()
                            synthesis_prompt = (
                                f"Web search results:\n{web_result}\n\n"
                                f"Query: {message}\n\n"
                                f"Provide a comprehensive answer based on these search results."
                            )
                            synthesized = llm.generate(
                                synthesis_prompt,
                                system="You are a helpful assistant. Answer based on the web search results provided.",
                            )
                            clickable = make_citations_clickable(synthesized, [])
                            chat_history.append({"role": "assistant", "content": clickable})
                            yield "", chat_history, "Sources: Web search results"
                            return
                        except Exception:
                            logger.exception("Web search fallback failed")

                    citations_display = _format_sources(result.sources)
                    clickable_response = make_citations_clickable(result.response, result.sources)
                    current_history = chat_history.copy()
                    current_history.append({"role": "assistant", "content": clickable_response})
                    yield "", current_history, citations_display

            # Only reached if no fallback was performed
            chat_history.append({"role": "assistant", "content": assistant_response})
            yield "", chat_history, citations_display

        except RateLimitError as e:
            logger.warning("Rate limit / capacity error: %s", e)
            chat_history.pop()
            error_message = (
                "The AI service is temporarily at capacity. Please wait a few moments and try again. "
                "If this keeps happening, try switching to a different model."
            )
            chat_history.append({"role": "assistant", "content": error_message})
            yield "", chat_history, citations_display
        except Exception:
            logger.exception("Error generating response")
            chat_history.pop()
            error_message = "I encountered an error while processing your query. Please try again."
            chat_history.append({"role": "assistant", "content": error_message})
            yield "", chat_history, citations_display

    with gr.Blocks(title="NagrikAI - Indian Immigration Assistant") as demo:
        session_state = gr.State(value=str(uuid4()))

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

        msg.submit(respond_stream, [msg, chatbot, session_state], [msg, chatbot, sources_panel])
        submit.click(respond_stream, [msg, chatbot, session_state], [msg, chatbot, sources_panel])
        clear.click(
            lambda: ([], "", "### Sources\n\nNo sources yet...", str(uuid4())),
            None,
            [chatbot, msg, sources_panel, session_state],
        )

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

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator
from typing import Any

import gradio as gr

from nagrik_ai.factories import create_orchestrator
from nagrik_ai.services.rag_orchestrator import RAGOrchestrator

logger = logging.getLogger(__name__)


def _build_ui(orch: RAGOrchestrator) -> gr.Blocks:
    """Build the Gradio interface used by the launcher."""
    return _build_streaming_ui(orch)


def _build_streaming_ui(orch: RAGOrchestrator) -> gr.Blocks:
    """Build a Gradio interface with streaming response support."""

    def respond_stream(
        message: str,
        chat_history: list[dict[str, str]],
    ) -> Generator[tuple[str, list[dict[str, str]]]]:
        if not chat_history:
            chat_history = []

        chat_history.append({"role": "user", "content": message})
        yield "", chat_history

        try:
            stream: Iterator[str] = orch.query_stream(message)
            assistant_response = ""
            for chunk in stream:
                assistant_response += chunk
                current_history = chat_history.copy()
                current_history.append({"role": "assistant", "content": assistant_response})
                yield "", current_history

            chat_history.append({"role": "assistant", "content": assistant_response})
            yield "", chat_history

        except Exception:
            logger.exception("Error generating response")
            error_message = "I encountered an error while processing your query. Please try again."
            chat_history.append({"role": "assistant", "content": error_message})
            yield "", chat_history

    with gr.Blocks(title="NagrikAI - Indian Immigration Assistant") as demo:
        gr.Markdown("# NagrikAI")
        gr.Markdown(
            "AI-powered Indian immigration assistant. "
            "Ask me about residence permits, social security, taxation, and more."
        )

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
            submit = gr.Button("Submit")
            clear = gr.Button("Clear")

        msg.submit(respond_stream, [msg, chatbot], [msg, chatbot])
        submit.click(respond_stream, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: ([], ""), None, [chatbot, msg])

        gr.Examples(  # type: ignore[call-arg]
            examples=[
                "How do I apply for a residence permit?",
                "What documents do I need for family reunification?",
                "How long does it take to process a work permit?",
                "What are the requirements for Indian citizenship?",
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

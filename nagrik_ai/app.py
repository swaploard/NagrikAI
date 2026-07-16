from __future__ import annotations

from typing import Any

import gradio as gr

from nagrik_ai.factories import create_orchestrator
from nagrik_ai.services.rag_orchestrator import RAGOrchestrator


def _build_ui(orch: RAGOrchestrator) -> gr.Blocks:
    def respond(message: str, _history: list[dict[str, object]]) -> str:
        return orch.query(message)

    with gr.Blocks(title="NagrikAI — Indian Immigration Assistant") as demo:
        gr.Markdown("# NagrikAI")
        gr.Markdown(
            "AI-powered Indian immigration assistant. "
            "Ask me about residence permits, social security, taxation, and more."
        )

        gr.ChatInterface(
            fn=respond,
            title="NagrikAI",
            description="Ask a question about Indian immigration.",
        )

    return demo  # type: ignore[no-any-return]


def launch(orch: RAGOrchestrator | None = None, **kwargs: Any) -> None:
    orch = orch or create_orchestrator()
    demo = _build_ui(orch)
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch()

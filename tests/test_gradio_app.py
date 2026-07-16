from __future__ import annotations

from unittest.mock import MagicMock

from nagrik_ai.app import _build_ui
from nagrik_ai.prompts.prompt_loader import PromptLoader
from nagrik_ai.services.rag_orchestrator import RAGOrchestrator


def test_build_ui_returns_blocks(mock_retrieval_service: MagicMock, mock_llm_service: MagicMock) -> None:
    prompt_loader = MagicMock(spec=PromptLoader)
    orch = RAGOrchestrator(
        retrieval_service=mock_retrieval_service,
        llm_service=mock_llm_service,
        prompt_loader=prompt_loader,
    )
    demo = _build_ui(orch)
    assert demo is not None
    assert hasattr(demo, "launch")

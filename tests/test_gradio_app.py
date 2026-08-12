from __future__ import annotations

from unittest.mock import MagicMock

from nagrik_ai.app import _build_ui


def test_build_ui_returns_blocks(mock_retrieval_service: MagicMock, mock_llm_service: MagicMock) -> None:
    demo = _build_ui()
    assert demo is not None
    assert hasattr(demo, "launch")

from __future__ import annotations

from pathlib import Path

import pytest

from nagrik_ai.prompts.prompt_registry import (
    CompiledPromptPipeline,
    PromptRegistry,
    PromptRegistryError,
    RenderedPrompt,
)


def test_prompt_registry_loads_default_pipeline() -> None:
    registry = PromptRegistry.load()

    pipeline = registry.get_pipeline()

    assert registry.default_pipeline == "gst_rag_default"
    assert pipeline.name == "gst_rag_default"
    assert pipeline.version == "1.0.0"
    assert [layer.name for layer in pipeline.layers] == [
        "system_prompt",
        "developer_prompt",
        "citation_policy",
        "context",
        "user_query",
    ]


def test_prompt_registry_compile_and_render() -> None:
    compiled = PromptRegistry.load().compile()

    rendered = compiled.render(
        context="[1]\nSource ID: s1\nContent:\nGST registration guidance",
        question="What is GST registration?",
        verbosity="concise",
        question_type="FACTUAL",
    )

    assert isinstance(compiled, CompiledPromptPipeline)
    assert isinstance(rendered, RenderedPrompt)
    assert rendered.pipeline_id == "gst_rag_default"
    assert rendered.pipeline_version == "1.0.0"
    assert "BEGIN AUTHORITATIVE GST SOURCES" in rendered.text
    assert "GST registration guidance" in rendered.text
    assert "QUESTION: What is GST registration?" in rendered.text
    assert len(rendered.prompt_content_hash) == 12


def test_prompt_registry_static_text_excludes_runtime_layers() -> None:
    compiled = PromptRegistry.load().compile()

    assert "You are an expert Indian GST" in compiled.static_text
    assert "$context" not in compiled.static_text
    assert "$question" not in compiled.static_text


def test_prompt_registry_missing_layer_fails(tmp_path: Path) -> None:
    (tmp_path / "pipelines.yaml").write_text(
        """
default_pipeline: broken
pipelines:
  broken:
    version: 1.0.0
    layers:
      - missing_prompt
""",
        encoding="utf-8",
    )

    with pytest.raises(PromptRegistryError, match="missing prompts"):
        PromptRegistry.load(tmp_path)

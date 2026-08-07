from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Template
from types import MappingProxyType

import yaml


class PromptRegistryError(Exception):
    """Raised when prompt registry configuration is invalid."""


@dataclass(frozen=True)
class PromptSpec:
    name: str
    file: Path
    content_hash: str


@dataclass(frozen=True)
class PromptPipeline:
    name: str
    version: str
    layers: tuple[PromptSpec, ...]


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    pipeline_id: str
    pipeline_version: str
    prompt_content_hash: str


@dataclass(frozen=True)
class _CompiledLayer:
    spec: PromptSpec
    template: str


@dataclass(frozen=True)
class CompiledPromptPipeline:
    pipeline: PromptPipeline
    layers: tuple[_CompiledLayer, ...]
    static_text: str

    def render(
        self,
        *,
        context: str = "",
        question: str = "",
        verbosity: str = "",
        question_type: str = "",
    ) -> RenderedPrompt:
        variables = {
            "context": context,
            "question": question,
            "verbosity": verbosity,
            "question_type": question_type,
        }
        rendered_layers = [Template(layer.template).safe_substitute(**variables).strip() for layer in self.layers]
        text = "\n\n".join(layer for layer in rendered_layers if layer)
        return RenderedPrompt(
            text=text,
            pipeline_id=self.pipeline.name,
            pipeline_version=self.pipeline.version,
            prompt_content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
        )


class PromptRegistry:
    def __init__(
        self,
        *,
        prompts_dir: Path,
        default_pipeline: str,
        pipelines: dict[str, tuple[str, tuple[str, ...]]],
    ) -> None:
        self._prompts_dir = prompts_dir
        self._default_pipeline = default_pipeline
        self._pipeline_defs = MappingProxyType(dict(pipelines))
        self._prompt_specs = MappingProxyType(self._load_prompt_specs())

    @classmethod
    def load(cls, prompts_dir: Path | None = None) -> PromptRegistry:
        resolved_prompts_dir = prompts_dir or Path(__file__).resolve().parent
        config_path = resolved_prompts_dir / "pipelines.yaml"
        if not config_path.exists():
            raise PromptRegistryError(f"Prompt pipeline config not found: {config_path}")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise PromptRegistryError("Prompt pipeline config must be a mapping")

        default_pipeline = raw.get("default_pipeline")
        raw_pipelines = raw.get("pipelines")
        if not isinstance(default_pipeline, str) or not default_pipeline:
            raise PromptRegistryError("Prompt pipeline config requires a default_pipeline string")
        if not isinstance(raw_pipelines, dict) or not raw_pipelines:
            raise PromptRegistryError("Prompt pipeline config requires at least one pipeline")

        pipelines: dict[str, tuple[str, tuple[str, ...]]] = {}
        for name, raw_pipeline in raw_pipelines.items():
            if not isinstance(name, str) or not isinstance(raw_pipeline, dict):
                raise PromptRegistryError("Each pipeline must be a mapping keyed by name")
            version = raw_pipeline.get("version")
            layers = raw_pipeline.get("layers")
            if not isinstance(version, str) or not version:
                raise PromptRegistryError(f"Pipeline {name!r} requires a version string")
            if not isinstance(layers, list) or not all(isinstance(layer, str) for layer in layers):
                raise PromptRegistryError(f"Pipeline {name!r} requires a list of layer names")
            pipelines[name] = (version, tuple(layers))

        if default_pipeline not in pipelines:
            raise PromptRegistryError(f"default_pipeline {default_pipeline!r} is not defined")

        registry = cls(
            prompts_dir=resolved_prompts_dir,
            default_pipeline=default_pipeline,
            pipelines=pipelines,
        )
        registry.validate()
        return registry

    @property
    def default_pipeline(self) -> str:
        return self._default_pipeline

    def _load_prompt_specs(self) -> dict[str, PromptSpec]:
        prompt_names = {layer for _, layers in self._pipeline_defs.values() for layer in layers}
        specs: dict[str, PromptSpec] = {}
        for name in prompt_names:
            path = self._prompts_dir / f"{name}.md"
            if not path.exists():
                continue
            content = path.read_bytes()
            specs[name] = PromptSpec(
                name=name,
                file=path,
                content_hash=hashlib.sha256(content).hexdigest()[:12],
            )
        return specs

    def validate(self, name: str | None = None) -> None:
        pipeline_name = name or self._default_pipeline
        if pipeline_name not in self._pipeline_defs:
            raise PromptRegistryError(f"Unknown prompt pipeline: {pipeline_name}")

        _, layers = self._pipeline_defs[pipeline_name]
        missing = [layer for layer in layers if layer not in self._prompt_specs]
        if missing:
            missing_list = ", ".join(missing)
            raise PromptRegistryError(f"Pipeline {pipeline_name!r} references missing prompts: {missing_list}")

    def get_prompt(self, name: str) -> PromptSpec:
        try:
            return self._prompt_specs[name]
        except KeyError as exc:
            raise PromptRegistryError(f"Unknown prompt: {name}") from exc

    def get_pipeline(self, name: str | None = None) -> PromptPipeline:
        pipeline_name = name or self._default_pipeline
        self.validate(pipeline_name)
        version, layer_names = self._pipeline_defs[pipeline_name]
        return PromptPipeline(
            name=pipeline_name,
            version=version,
            layers=tuple(self.get_prompt(layer_name) for layer_name in layer_names),
        )

    def compile(self, name: str | None = None) -> CompiledPromptPipeline:
        pipeline = self.get_pipeline(name)
        layers = tuple(
            _CompiledLayer(
                spec=spec,
                template=spec.file.read_text(encoding="utf-8"),
            )
            for spec in pipeline.layers
        )
        static_text = "\n\n".join(
            layer.template.strip() for layer in layers if layer.spec.name not in {"context", "user_query"}
        )
        return CompiledPromptPipeline(pipeline=pipeline, layers=layers, static_text=static_text)


def load_default_prompt_pipeline() -> CompiledPromptPipeline:
    return PromptRegistry.load().compile()

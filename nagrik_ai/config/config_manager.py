from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from nagrik_ai.config.config_models import NagrikAIConfig


class ConfigManager:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or Path(__file__).parent / "site_configs.yaml"
        self._config: NagrikAIConfig | None = None

    def load(self) -> NagrikAIConfig:
        if self._config is not None:
            return self._config

        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        try:
            self._config = NagrikAIConfig(**raw)
        except ValidationError as e:
            msg = f"Invalid configuration in {self.config_path}"
            raise ValueError(msg) from e
        return self._config

    @property
    def config(self) -> NagrikAIConfig:
        if self._config is None:
            return self.load()
        return self._config

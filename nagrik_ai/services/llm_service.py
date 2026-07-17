from __future__ import annotations

import logging
from collections.abc import Iterator

from ollama import Client

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2") -> None:
        self.client = Client(host=base_url)
        self.model = model
        logger.info("Initialized LLM service with model: %s", model)

    def generate(self, prompt: str, system: str | None = None) -> str:
        options: dict[str, str] = {}
        if system:
            options["system"] = system
        response = self.client.generate(model=self.model, prompt=prompt, options=options)
        return response.response or ""

    def generate_stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        options: dict[str, str] = {}
        if system:
            options["system"] = system
        stream = self.client.generate(model=self.model, prompt=prompt, options=options, stream=True)
        for chunk in stream:
            yield chunk.response or ""

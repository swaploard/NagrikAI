from __future__ import annotations

from pathlib import Path
from string import Template


class PromptLoader:
    def __init__(self, prompt_dir: Path | None = None) -> None:
        self.prompt_dir = prompt_dir or Path(__file__).parent

    def load_system(self) -> str:
        path = self.prompt_dir / "system_prompt.md"
        return path.read_text(encoding="utf-8")

    def load_user(self, query: str, context: str) -> str:
        path = self.prompt_dir / "user_query.md"
        template = Template(path.read_text(encoding="utf-8"))
        return template.safe_substitute(query=query, context=context)

# NagrikAI — Project Context for Claude Code

## Commands
- `uv run ruff check .` — Lint all files
- `uv run ruff format .` — Format all files
- `uv run mypy nagrik_ai` — Typecheck the package
- `uv run pytest` — Run all tests
- `uv run pytest tests/test_file.py -k test_name -v` — Run specific test
- `uv run nagrik-ai --help` — CLI help
- `uv run nagrik-ai crawl sites` — Crawl all configured sites
- `uv run nagrik-ai parse all` — Parse crawled HTML to Markdown
- `uv run nagrik-ai vectorize run` — Vectorize parsed docs to ChromaDB
- `uv run nagrik-ai app-command` — Launch the Gradio UI

## Architecture
- RAG pipeline: Crawl → Parse → Vectorize → ChromaDB → Retrieve → LLM → Response
- Config-driven: sites defined in `nagrik_ai/config/site_configs.yaml`
- DI via `nagrik_ai/factories.py` — swap out ChromaStore, LLMService, etc.
- Prompts in `nagrik_ai/prompts/` use `string.Template` for variable substitution

## Code style
- Python 3.14+, strict type annotations everywhere
- Ruff linting with 120+ rules selected (see pyproject.toml)
- Double quotes for strings
- f-strings preferred over `.format()` or `%`
- No comments in implementation code unless necessary for clarity

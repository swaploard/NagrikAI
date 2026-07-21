# NagrikAI

AI-powered Indian GST assistant using a RAG pipeline over official sources.

### Commands

Examples:

```bash
uv run nagrik-ai crawl sites                     # crawl all sites at depth 1
uv run nagrik-ai parse all                       # parse all sites under content/
uv run nagrik-ai vectorize                       # vectorise all markdown files
uv run nagrik-ai app-command
uv run pytest tests/ -v
```

### Direct Tool Examples

`example/03_direct_tools.py` invokes each real tool directly (no LLM router), one demo per tool:

```bash
uv run python example/03_direct_tools.py --pdf            # self-contained, no env/data
uv run python example/03_direct_tools.py --rag            # real ChromaDB
uv run python example/03_direct_tools.py --web            # real Tavily
uv run python example/03_direct_tools.py --all            # all three
uv run python example/03_direct_tools.py --all --synthesize
uv run python example/03_direct_tools.py --pdf --pdf-path /path/to/file.pdf
```

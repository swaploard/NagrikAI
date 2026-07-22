# NagrikAI

AI-powered Indian GST assistant using a RAG pipeline over official sources.

### Why NagrikAI

- **Official source fidelity** – Crawls only authoritative Indian government sites (india.gov.in, gst.gov.in, tutorial.gst.gov.in) ensuring accurate, up-to-date information
- **Hybrid retrieval** – Combines BM25 keyword search with BGE-M3 embeddings and BGE-Reranker-Large reranking for precise, relevant results
- **Transparent reasoning** – Every answer includes verifiable citations tracing back to specific source documents
- **Fallback resilience** – When RAG yields insufficient context, automatically falls back to web search with LLM synthesis
- **Local-first design** – Defaults to Ollama for fully private, offline-capable operation; OpenRouter available for cloud LLMs
- **Observability built-in** – Full LangSmith tracing for monitoring, debugging, and continuous improvement
- **Type-safe foundation** – Built on Python 3.14 with Pydantic, MyPy, and Ruff for robust, maintainable code
- **Modular architecture** – Clean separation of crawling, parsing, vectorization, retrieval, and generation concerns

### No embeddings storage, on purpose

Unlike many RAG systems that store dense vector embeddings permanently, NagrikAI computes embeddings during the vectorization phase and stores only the BM25 index and document metadata in ChromaDB. Embeddings are generated on-demand during retrieval using sentence-transformers. This approach:

- Eliminates storage overhead of high-dimensional vectors (BGE-M3 uses 1024-dim vectors)
- Avoids embedding staleness when models are updated
- Simplifies the pipeline – update the embedding model by re-running vectorization
- Reduces complexity while maintaining retrieval quality through hybrid search

### Bounded context, on purpose

NagrikAI operates with a finite context window determined by the LLM's capacity (typically 8K-32K tokens). Rather than attempting to retrieve and reason over unlimited documents, we:

- Retrieve top-K chunks (configurable, default 5) after reranking from Fetch-K candidates (default 20)
- Apply strict token limits to context construction
- Truncate overly long documents during parsing
- Accept that some niche queries may require refinement or fallback to web search

This constraint ensures predictable latency, prevents context overflow errors, and focuses the LLM on the most relevant information – mirroring how human researchers work with limited cognitive bandwidth.

### Installation

```bash
# Clone and install
git clone https://github.com/your-org/nagrik-ai.git
cd nagrik-ai

# Install with UV (recommended)
uv sync

# Or with pip
pip install -e .

# Verify installation
uv run nagrik-ai --help
```

### Quick start

```bash
# 1. Crawl official government sources (runs scrapy spiders)
uv run nagrik-ai crawl sites

# 2. Parse HTML to Markdown
uv run nagrik-ai parse all

# 3. Vectorize documents into ChromaDB
uv run nagrik-ai vectorize

# 4. Launch the Gradio web interface
uv run nagrik-ai app-command

# Optional: Use OpenRouter instead of local Ollama
uv run nagrik-ai app-command --llm-provider openrouter
```

### Commands

| Command                                | Description                               |
| -------------------------------------- | ----------------------------------------- |
| `nagrik-ai crawl sites`                | Crawl all configured government sites     |
| `nagrik-ai parse all`                  | Convert crawled HTML to clean Markdown    |
| `nagrik-ai vectorize`                  | Generate embeddings and populate ChromaDB |
| `nagrik-ai app-command`                | Launch the Gradio web UI                  |
| `nagrik-ai trace test "Your question"` | Test with LangSmith tracing enabled       |
| `uv run pytest tests/ -v`              | Run the test suite                        |

### Shell integration

Add this to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) for quick access:

```bash
# Quick NagrikAI access
alias nagrik='uv run nagrik-ai app-command'
```

Then simply type `nagrik` in your terminal to launch the assistant.

### Supported providers

| Provider             | Setup                                         | Notes                          |
| -------------------- | --------------------------------------------- | ------------------------------ |
| **Ollama** (default) | `ollama serve` + `ollama pull mistral:latest` | Fully private, local inference |
| **OpenRouter**       | Set `NAGRIKAI_OPENROUTER_API_KEY` env var     | Access to 100+ cloud LLMs      |

### Example configuration

Create .env file (or set environment variables):

```env
# LLM Provider (ollama or openrouter)
NAGRIKAI_LLM_PROVIDER=ollama

# Ollama settings (used when provider=ollama)
NAGRIKAI_OLLAMA_BASE_URL=http://localhost:11434
NAGRIKAI_OLLAMA_MODEL=mistral:latest

# OpenRouter settings (used when provider=openrouter)
NAGRIKAI_OPENROUTER_API_KEY=your-key-here
NAGRIKAI_OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# Embedding and reranking
NAGRIKAI_EMBEDDING_MODEL=BAAI/bge-m3
NAGRIKAI_RERANKER_MODEL=BAAI/bge-reranker-large
NAGRIKAI_RERANKER_ENABLED=true
NAGRIKAI_HYBRID_SEARCH_ENABLED=true

# Retrieval tuning
NAGRIKAI_TOP_K=5
NAGRIKAI_FETCH_K=20
NAGRIKAI_LAMBDA_MULT=0.7
NAGRIKAI_BM25_K1=1.5
NAGRIKAI_BM25_B=0.75
NAGRIKAI_RRF_K=60

# Observability
LANGSMITH_TRACING_ENABLED=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=nagrik-ai
```

### License

MIT

### Acknowledgements

- [LangChain](https://www.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/) for LLM orchestration
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) via [rank-bm25](https://pypi.org/project/rank-bm25/) for keyword search
- [Gradio](https://www.gradio.app/) for the web interface
- [Typer](https://tiangolo.com/typer/) for the CLI
- [Pydantic](https://docs.pydantic.dev/) for configuration management
- [Ruff](https://docs.astral.sh/ruff/) and [MyPy](https://mypy-lang.org/) for code quality
- Official Indian government websites (india.gov.in, gst.gov.in, tutorial.gst.gov.in) as knowledge sources

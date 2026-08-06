from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, HttpUrl

load_dotenv()


def _get_env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _get_env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    return int(value) if value is not None else default


def _get_env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    return float(value) if value is not None else default


def _get_env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in {"true", "1", "yes", "on"}


class CrawlerConfig(BaseModel):
    max_concurrency: int = 5
    request_delay: float = 1.0
    timeout: int = 30
    max_depth: int = 1
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )


class MarkdownConfig(BaseModel):
    ignore_links: bool = False
    body_width: int = 0
    protect_links: bool = True
    unicode_snob: bool = True
    ignore_images: bool = False
    ignore_tables: bool = False


class ParserConfig(BaseModel):
    content_selector: str = "main"
    title_selector: str = "//title"
    strip_selectors: list[str] = ["nav", "footer", "script", "style"]
    fallback_to_body: bool = True
    output_format: str = "markdown"
    markdown: MarkdownConfig = MarkdownConfig()


class SiteConfig(BaseModel):
    name: str
    base_url: HttpUrl
    start_urls: list[HttpUrl]
    allowed_domains: list[str]
    crawler: CrawlerConfig = CrawlerConfig()
    parser: ParserConfig = ParserConfig()


class OpenRouterConfig(BaseModel):
    api_key: str = _get_env("NAGRIKAI_OPENROUTER_API_KEY", "")
    base_url: str = _get_env("NAGRIKAI_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model: str = _get_env("NAGRIKAI_OPENROUTER_MODEL", "openrouter/auto")


class EvaluationConfig(BaseModel):
    langsmith_project: str = "nagrik-ai-eval"
    langsmith_api_key: str = _get_env("NAGRIKAI_LANGSMITH_API_KEY", "")
    langsmith_endpoint: str = _get_env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    langsmith_tracing_enabled: bool = _get_env_bool("LANGSMITH_TRACING_ENABLED", True)
    langsmith_trace_verbose: bool = _get_env_bool("LANGSMITH_TRACE_VERBOSE", False)
    deepeval_dataset: str = "nagrik-ai-golden"
    ragas_dataset: str = "nagrik-ai-golden"
    golden_dataset_path: str = "data/golden_dataset.jsonl"
    experiment_prefix: str = "rag-eval"
    judge_llm_provider: str = _get_env("NAGRIKAI_JUDGE_LLM_PROVIDER", "openrouter")
    judge_llm_model: str = _get_env("NAGRIKAI_JUDGE_LLM_MODEL", "anthropic/claude-3.5-sonnet")


class NagrikAIConfig(BaseModel):
    sites: list[SiteConfig]
    chroma_persist_dir: str = _get_env("NAGRIKAI_CHROMA_PERSIST_DIR", "chroma_db")
    content_dir: str = _get_env("NAGRIKAI_CONTENT_DIR", "content")
    embedding_model: str = _get_env("NAGRIKAI_EMBEDDING_MODEL", "BAAI/bge-m3")
    llm_provider: str = _get_env("NAGRIKAI_LLM_PROVIDER", "ollama")
    ollama_base_url: str = _get_env("NAGRIKAI_OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = _get_env("NAGRIKAI_OLLAMA_MODEL", "qwen2.5:7b")
    openrouter: OpenRouterConfig = OpenRouterConfig()
    chunk_size: int = _get_env_int("NAGRIKAI_CHUNK_SIZE", 800)
    chunk_overlap: int = _get_env_int("NAGRIKAI_CHUNK_OVERLAP", 200)
    top_k: int = _get_env_int("NAGRIKAI_TOP_K", 5)
    fetch_k: int = _get_env_int("NAGRIKAI_FETCH_K", 20)
    lambda_mult: float = _get_env_float("NAGRIKAI_LAMBDA_MULT", 0.7)
    reranker_model: str = _get_env("NAGRIKAI_RERANKER_MODEL", "BAAI/bge-reranker-large")
    reranker_enabled: bool = _get_env_bool("NAGRIKAI_RERANKER_ENABLED", True)
    hybrid_search_enabled: bool = _get_env_bool("NAGRIKAI_HYBRID_SEARCH_ENABLED", True)
    bm25_k1: float = _get_env_float("NAGRIKAI_BM25_K1", 1.5)
    bm25_b: float = _get_env_float("NAGRIKAI_BM25_B", 0.75)
    rrf_k: int = _get_env_int("NAGRIKAI_RRF_K", 60)
    max_response_tokens: int = _get_env_int("NAGRIKAI_MAX_RESPONSE_TOKENS", 600)
    authority_ranking_enabled: bool = _get_env_bool("NAGRIKAI_AUTHORITY_RANKING_ENABLED", True)
    authority_bonus: dict[str, float] = {
        "act": 0.08,
        "rules": 0.08,
        "notification": 0.06,
        "circular": 0.05,
        "faq": 0.02,
        "user_guide": 0.01,
        "help": 0.0,
        "other": 0.0,
    }
    checkpoint_dir: str = _get_env("NAGRIKAI_CHECKPOINT_DIR", "checkpoints")


_defaults = NagrikAIConfig.model_construct(sites=[])

CHROMA_PERSIST_DIR = Path(str(_defaults.chroma_persist_dir))
CONTENT_DIR = Path(str(_defaults.content_dir))
EMBEDDING_MODEL = _defaults.embedding_model
LLM_PROVIDER = _defaults.llm_provider
OLLAMA_BASE_URL = _defaults.ollama_base_url
OLLAMA_MODEL = _defaults.ollama_model
OPENROUTER_API_KEY = _defaults.openrouter.api_key
OPENROUTER_BASE_URL = _defaults.openrouter.base_url
OPENROUTER_MODEL = _defaults.openrouter.model
CHUNK_SIZE = _defaults.chunk_size
CHUNK_OVERLAP = _defaults.chunk_overlap
TOP_K = _defaults.top_k
FETCH_K = _defaults.fetch_k
LAMBDA_MULT = _defaults.lambda_mult
RERANKER_MODEL = _defaults.reranker_model
RERANKER_ENABLED = _defaults.reranker_enabled
HYBRID_SEARCH_ENABLED = _defaults.hybrid_search_enabled
BM25_K1 = _defaults.bm25_k1
BM25_B = _defaults.bm25_b
RRF_K = _defaults.rrf_k
AUTHORITY_RANKING_ENABLED = _defaults.authority_ranking_enabled
AUTHORITY_BONUS = dict(_defaults.authority_bonus)
MAX_RESPONSE_TOKENS = _defaults.max_response_tokens
CHECKPOINT_DIR = Path(str(_defaults.checkpoint_dir))

EVAL_CONFIG = EvaluationConfig()

LANGSMITH_PROJECT = EVAL_CONFIG.langsmith_project
LANGSMITH_API_KEY = EVAL_CONFIG.langsmith_api_key
LANGSMITH_ENDPOINT = EVAL_CONFIG.langsmith_endpoint
LANGSMITH_TRACING_ENABLED = EVAL_CONFIG.langsmith_tracing_enabled
LANGSMITH_TRACE_VERBOSE = EVAL_CONFIG.langsmith_trace_verbose

del _defaults

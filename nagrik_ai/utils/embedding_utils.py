from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    model = load_embedding_model(model_name)
    result = model.encode(text)
    return result.tolist()


def embed_batch(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    model = load_embedding_model(model_name)
    result = model.encode(texts)
    return result.tolist()  # type: ignore[no-any-return]

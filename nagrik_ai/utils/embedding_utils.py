"""Utilities for generating embeddings using LangChain."""

import logging
from typing import cast

from langchain_community.embeddings import (
    SentenceTransformerEmbeddings,
)

from nagrik_ai.config.config_models import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using LangChain's SentenceTransformerEmbeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Initialized embedding model: %s", model_name)

    def generate(self, text: str) -> list[float] | None:
        try:
            embedding = self.embedding_model.embed_query(text)
        except Exception:
            logger.exception("Error generating embedding")
            return None
        else:
            return embedding

    def generate_batch(self, texts: list[str]) -> list[list[float] | None]:
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            return cast("list[list[float] | None]", embeddings)
        except Exception:
            logger.exception("Error generating batch embeddings")
            return [None for _ in texts]

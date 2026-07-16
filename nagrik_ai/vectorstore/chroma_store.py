from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document as LangChainDocument
from langchain_core.vectorstores import VectorStoreRetriever


class ChromaStore:
    def __init__(self, persist_dir: str | Path, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.persist_dir = str(persist_dir)
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self._store: Chroma | None = None

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
        return self._store

    def add_texts(self, texts: list[str], metadatas: list[dict[str, object]] | None = None) -> list[str]:
        return self.store.add_texts(texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 5) -> list[LangChainDocument]:
        return self.store.similarity_search(query, k=k)

    def as_retriever(self, k: int = 5) -> VectorStoreRetriever:
        return self.store.as_retriever(search_kwargs={"k": k})

    def count(self) -> int:
        return self.store._collection.count()

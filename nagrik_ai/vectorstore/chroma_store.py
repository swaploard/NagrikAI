"""ChromaDB vector store abstraction using LangChain."""

import logging
from typing import Any, cast

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

STANDARD_METADATA_FIELDS: dict[str, type] = {
    "title": str,
    "url": str,
    "domain": str,
    "source_id": str,
    "source_type": str,
    "chunk_index": int,
    "total_chunks": int,
    "score": float,
}


def validate_metadata(meta: dict[Any, Any]) -> dict[str, Any]:
    """Fill missing fields with safe defaults; prevent NoneType crashes."""
    logger.debug("Validating metadata with keys: %s", list(meta.keys()))
    validated: dict[str, Any] = {}
    for key, typ in STANDARD_METADATA_FIELDS.items():
        value = meta.get(key)
        if value is None:
            logger.debug("Metadata field '%s' missing, using default for type %s", key, typ.__name__)
            value = 0.0 if typ is float else (0 if typ is int else "")
        try:
            validated[key] = typ(value) if not isinstance(value, typ) else value
        except (ValueError, TypeError) as e:
            logger.warning(
                "Failed to cast metadata field '%s' (value=%r) to %s: %s, using default",
                key,
                value,
                typ.__name__,
                e,
            )
            validated[key] = 0.0 if typ is float else (0 if typ is int else "")
    logger.debug("Validated metadata keys: %s", list(validated.keys()))
    return validated


class ChromaStore:
    """LangChain-based ChromaDB vector store abstraction.

    This class wraps LangChain's Chroma implementation with a simpler interface.
    Dependencies are injected to allow for testing and configuration flexibility.
    """

    def __init__(
        self,
        collection_name: str,
        embeddings: Embeddings,
        persist_directory: str = "chroma_db",
    ) -> None:
        """Initialize the ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            embeddings: Embeddings instance (e.g., HuggingFaceEmbeddings)
            persist_directory: Directory to persist the ChromaDB database

        Example:
            >>> from langchain_huggingface import HuggingFaceEmbeddings
            >>> embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            >>> store = ChromaStore(
            ...     collection_name="my_docs",
            ...     embeddings=embeddings,
            ...     persist_directory="./chroma_db"
            ... )
        """
        self.embeddings = embeddings

        # Initialize the vector store
        self.vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        logger.info("Initialized ChromaStore with collection: %s", collection_name)

    def add_document(
        self,
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a document to the vector store.

        This method is maintained for backward compatibility but uses the langchain Chroma
        implementation internally.

        Args:
            document_id: Unique identifier for the document
            metadata: Document metadata
        """
        try:
            metadata_dict = metadata if metadata is not None else {}

            # Extract content from metadata if available
            document_text = metadata_dict.get("content", "")

            # If content is missing from metadata but available elsewhere, try to find it
            if not document_text:
                # Look for content in other common field names
                for field in ["text", "body", "page_content", "full_text"]:
                    if field in metadata_dict:
                        document_text = metadata_dict[field]
                        break

            # Ensure we have some text content
            if not document_text:
                logger.warning("No content found for document %s", document_id)
                document_text = f"Empty document: {document_id}"

            # Create a LangChain Document and add via add_documents to ensure proper typing
            # Note: Chroma will compute embeddings automatically if not provided
            validated_meta = validate_metadata(metadata_dict)
            doc = Document(page_content=document_text, metadata=validated_meta)
            added_ids = self.vector_db.add_documents([doc])
            # If the underlying store returned ids, ensure the requested id is recorded in metadata
            if added_ids and document_id:
                # try to set chroma id in metadata for consistency
                metadata_dict.setdefault("chroma_id", document_id)
            _ = added_ids

            logger.debug("Added document %s to vector store", document_id)

        except Exception:
            logger.exception("Failed to add document %s", document_id)
            raise

    def similarity_search(
        self,
        query_text: str,
        k: int = 20,
    ) -> list[Document]:
        """Query the vector store by text using plain similarity search.

        Args:
            query_text: Text to search for
            k: Number of nearest neighbors to return

        Returns:
            List of documents most similar to the query
        """
        try:
            results: list[Document] = self.vector_db.similarity_search(
                query=query_text,
                k=k,
            )

            for doc in results:
                self._enhance_document_with_citation(doc)
                raw_metadata: dict[str, Any] | None = getattr(doc, "metadata", None)
                if isinstance(raw_metadata, dict):
                    doc.metadata = validate_metadata(raw_metadata)
        except Exception:
            logger.exception("Failed to similarity search vector store")
            return []
        else:
            return results

    def similarity_search_with_scores(
        self,
        query_text: str,
        k: int = 20,
    ) -> list[tuple[Document, float]]:
        """Query the vector store and return docs paired with relevance scores.

        Args:
            query_text: Text to search for
            k: Number of nearest neighbors to return

        Returns:
            List of (document, relevance score) pairs, score in [0, 1].
        """
        try:
            results: list[tuple[Document, float]] = self.vector_db.similarity_search_with_relevance_scores(
                query=query_text,
                k=k,
            )
            for doc, _score in results:
                self._enhance_document_with_citation(doc)
                raw_metadata: dict[str, Any] | None = getattr(doc, "metadata", None)
                if isinstance(raw_metadata, dict):
                    doc.metadata = validate_metadata(raw_metadata)
        except Exception:
            logger.exception("Failed to similarity search vector store with scores")
            return []
        else:
            return results

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
    ) -> list[Document]:
        """Query the vector store by text using MMR for diversity.

        Args:
            query_text: Text to search for
            n_results: Number of results to return
            fetch_k: Number of candidates to fetch before MMR re-ranking
            lambda_mult: MMR diversity parameter (0 = max diversity, 1 = max relevance)

        Returns:
            List of documents most relevant and diverse for the query
        """
        try:
            results: list[Document] = self.vector_db.max_marginal_relevance_search(
                query=query_text,
                k=n_results,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )

            # Enhance results with citation information
            for doc in results:
                self._enhance_document_with_citation(doc)
                raw_metadata: dict[str, Any] | None = getattr(doc, "metadata", None)
                if isinstance(raw_metadata, dict):
                    doc.metadata = validate_metadata(raw_metadata)
        except Exception:
            logger.exception("Failed to query vector store")
            return []
        else:
            return results

    def query_with_embedding(
        self,
        embedding: list[float],
        n_results: int = 5,
    ) -> list[tuple[Document, float]]:
        """Query the vector store by embedding.

        Args:
            embedding: Embedding vector to search with
            n_results: Number of results to return

        Returns:
            List of (document, relevance score) pairs most similar to the query
        """
        try:
            results = self.vector_db.similarity_search_by_vector_with_relevance_scores(
                embedding,
                k=n_results,
            )

            # Enhance results with citation information
            for doc, _score in results:
                raw_metadata: dict[str, Any] | None = getattr(doc, "metadata", None)
                if isinstance(raw_metadata, dict):
                    doc.metadata = validate_metadata(raw_metadata)
        except Exception:
            logger.exception("Failed to query vector store with embedding")
            return []
        else:
            return results

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Get a document by ID.

        Args:
            document_id: ID of the document to retrieve

        Returns:
            The document if found
        """
        try:
            result: dict[str, Any] = self.vector_db.get(
                ids=[document_id],
                include=["documents", "metadatas"],
            )

            # Add citation information
            if result and "metadatas" in result and result["metadatas"]:
                for i, metadata in enumerate(result["metadatas"]):
                    if isinstance(metadata, dict):
                        metadata_typed = cast(dict[str, Any], metadata)
                        validated_metadata = validate_metadata(metadata_typed)
                        result["metadatas"][i] = validated_metadata

        except Exception:
            logger.exception("Failed to get document %s", document_id)
            return None
        else:
            return result

    def get_all_documents(self) -> list[Document]:
        """Get all documents from the vector store.

        Returns:
            List of all Document objects in the store
        """
        try:
            result = self.vector_db.get(include=["documents", "metadatas"])
            documents: list[Document] = []
            ids = result.get("ids", [])
            texts = result.get("documents", [])
            metadatas = result.get("metadatas", [])
            for doc_id, text, metadata in zip(ids, texts, metadatas, strict=False):
                md: dict[str, Any] = dict(metadata) if metadata else {}
                md["chroma_id"] = doc_id
                md = validate_metadata(md)
                doc = Document(page_content=text or "", metadata=md)
                self._enhance_document_with_citation(doc)
                documents.append(doc)
            return documents
        except Exception:
            logger.exception("Failed to get all documents")
            return []

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects to add

        Returns:
            List of document IDs
        """
        try:
            for doc in documents:
                raw_metadata: dict[str, Any] = getattr(doc, "metadata", {})
                doc.metadata = validate_metadata(raw_metadata)
            return self.vector_db.add_documents(documents)
        except Exception:
            logger.exception("Failed to add documents")
            raise

    def _enhance_document_with_citation(self, doc: Any) -> None:
        """Enhance a document with citation information.

        Args:
            doc: LangChain document object to enhance
        """
        if not hasattr(doc, "metadata"):
            return

        md = dict(doc.metadata)

        if "source_url" in md:
            md["citation_url"] = md["source_url"]
        elif "url" in md:
            md["citation_url"] = md["url"]

        doc.metadata = md

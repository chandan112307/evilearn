"""Vector storage module using ChromaDB for semantic retrieval.

Uses ChromaDB's built-in SentenceTransformerEmbeddingFunction for embedding
generation, so ChromaDB handles both storage and embedding natively.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import Optional


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB.

    ChromaDB handles embedding generation internally via
    SentenceTransformerEmbeddingFunction (all-MiniLM-L6-v2).
    """

    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        """Initialize ChromaDB persistent client.

        Args:
            persist_directory: Directory for persistent storage.
            model_name: Sentence transformer model for embeddings.
        """
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
        )
        self._collection_name = "evilearn_documents"
        self._collection: Optional[chromadb.Collection] = None

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the document collection."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embedding_function,
            )
        return self._collection

    def add_chunks(
        self,
        chunk_ids: list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Store chunks in ChromaDB. Embeddings are generated automatically.

        Args:
            chunk_ids: Unique IDs for each chunk.
            documents: Raw text of each chunk.
            metadatas: Metadata dicts (page_number, document_id) for each chunk.
        """
        self.collection.add(
            ids=chunk_ids,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve top-k similar chunks using text query.

        ChromaDB handles embedding the query text internally.

        Args:
            query_text: Claim text to search for.
            top_k: Number of results to return.
            document_id: Optional filter by document.

        Returns:
            List of evidence dicts with text_snippet, page_number, relevance_score.
        """
        where_filter = None
        if document_id:
            where_filter = {"document_id": document_id}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        evidence_list = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                relevance_score = max(0.0, 1.0 - distance)
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                evidence_list.append({
                    "text_snippet": doc,
                    "page_number": metadata.get("page_number", 0),
                    "relevance_score": round(relevance_score, 4),
                    "document_id": metadata.get("document_id", ""),
                })

        return evidence_list

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document.

        Args:
            document_id: ID of document to delete.
        """
        self.collection.delete(where={"document_id": document_id})

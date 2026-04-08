"""Vector storage module using ChromaDB for storage and similarity search ONLY.

ChromaDB does NOT generate embeddings. All embeddings are generated externally
by the EmbeddingService (via the LLM API) and passed into ChromaDB as
pre-computed vectors.

ChromaDB = storage + similarity search
LLM API  = embedding generation (via EmbeddingService)
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Optional


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB.

    ChromaDB stores and retrieves pre-computed embeddings.
    It NEVER generates embeddings on its own.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB persistent client.

        Args:
            persist_directory: Directory for persistent storage.
        """
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection_name = "evilearn_documents"
        self._collection: Optional[chromadb.Collection] = None

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the document collection.

        No embedding function is assigned — ChromaDB never auto-embeds.
        """
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(
        self,
        chunk_ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks with pre-computed embeddings in ChromaDB.

        Embeddings MUST be generated externally (via EmbeddingService)
        and passed in here. ChromaDB does NOT embed anything.

        Args:
            chunk_ids: Unique IDs for each chunk.
            documents: Raw text of each chunk.
            metadatas: Metadata dicts (page_number, document_id) for each chunk.
            embeddings: Pre-computed embedding vectors from EmbeddingService.
        """
        self.collection.add(
            ids=chunk_ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve top-k similar chunks using a pre-computed query embedding.

        The query embedding MUST be generated externally (via EmbeddingService).
        ChromaDB only performs the similarity search.

        Args:
            query_embedding: Pre-computed embedding vector for the query.
            top_k: Number of results to return.
            document_id: Optional filter by document.

        Returns:
            List of evidence dicts with text_snippet, page_number, relevance_score.
        """
        where_filter = None
        if document_id:
            where_filter = {"document_id": document_id}

        results = self.collection.query(
            query_embeddings=[query_embedding],
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

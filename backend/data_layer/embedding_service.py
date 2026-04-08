"""Embedding Service — generates embeddings via LLM API (OpenAI/Groq).

ChromaDB is used ONLY for storage and similarity search.
Embeddings MUST be generated externally using this service
BEFORE storing or querying ChromaDB.

Supported providers:
    - OpenAI: uses text-embedding-ada-002 or configurable model
    - Groq: uses the same OpenAI-compatible embeddings API
"""

import os
from typing import Optional


class EmbeddingService:
    """Generates embeddings using the LLM API.

    All embedding generation goes through this service.
    ChromaDB never generates its own embeddings.
    """

    def __init__(self, llm_client=None, provider: str = "groq"):
        """Initialize embedding service.

        Args:
            llm_client: OpenAI or Groq client instance.
            provider: LLM provider name ("openai" or "groq").
        """
        self._llm_client = llm_client
        self._provider = provider
        self._model = os.environ.get(
            "EMBEDDING_MODEL",
            "text-embedding-ada-002" if provider == "openai" else "text-embedding-ada-002",
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            RuntimeError: If embedding generation fails.
        """
        if not texts:
            return []

        if not self._llm_client:
            raise RuntimeError(
                "Embedding generation requires an LLM client. "
                "Set LLM_API_KEY environment variable."
            )

        try:
            response = self._llm_client.embeddings.create(
                model=self._model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")

    def embed_query(self, query_text: str) -> list[float]:
        """Generate an embedding for a single query string.

        Args:
            query_text: Text to embed.

        Returns:
            Embedding vector (list of floats).

        Raises:
            RuntimeError: If embedding generation fails.
        """
        embeddings = self.embed_texts([query_text])
        return embeddings[0]

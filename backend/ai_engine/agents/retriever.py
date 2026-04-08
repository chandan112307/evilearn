"""Retrieval Agent — Retrieves document evidence for each claim."""

from typing import Optional


class RetrievalAgent:
    """Retrieves relevant document chunks for each claim using vector similarity.

    Uses ONLY retrieved documents as evidence source.
    Returns top-k relevant chunks with page numbers.
    ChromaDB handles embedding generation internally.
    """

    def __init__(self, vector_store):
        """Initialize with vector store.

        Args:
            vector_store: VectorStore instance for similarity search.
        """
        self.vector_store = vector_store

    def run(self, claims: list[dict], top_k: int = 5) -> dict:
        """Retrieve evidence for each claim.

        Args:
            claims: List of claim dicts with claim_id and claim_text.
            top_k: Number of evidence chunks to retrieve per claim.

        Returns:
            Dict mapping claim_id to list of evidence objects.
        """
        evidence_map = {}

        for claim in claims:
            claim_id = claim["claim_id"]
            claim_text = claim["claim_text"]

            try:
                evidence = self.vector_store.query(
                    query_text=claim_text,
                    top_k=top_k,
                )
                evidence_map[claim_id] = evidence
            except Exception:
                evidence_map[claim_id] = []

        return evidence_map

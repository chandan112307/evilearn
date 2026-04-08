"""Text chunking module for splitting documents into retrievable units."""

import uuid


class TextChunker:
    """Splits document text into semantic chunks while preserving page mapping."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize chunker with size parameters.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[dict], document_id: str) -> list[dict]:
        """Split pages into chunks preserving page mapping.

        Args:
            pages: List of dicts with 'page_number' and 'text'.
            document_id: ID of the source document.

        Returns:
            List of chunk dicts with chunk_id, chunk_text, page_number, document_id.
        """
        chunks = []
        for page in pages:
            page_chunks = self._split_text(page["text"])
            for chunk_text in page_chunks:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_text": chunk_text,
                    "page_number": page["page_number"],
                    "document_id": document_id,
                })
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks with overlap.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)
                break_point = max(last_period, last_newline)
                if break_point > start:
                    end = break_point + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

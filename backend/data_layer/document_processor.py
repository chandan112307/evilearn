"""Document processing module using PyMuPDF for PDF text extraction."""

import fitz  # PyMuPDF
import uuid
from typing import Optional


class DocumentProcessor:
    """Processes uploaded documents and extracts text with page mapping."""

    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> list[dict]:
        """Extract text from PDF preserving page numbers.

        Args:
            file_bytes: Raw PDF file bytes.

        Returns:
            List of dicts with 'page_number' and 'text' keys.

        Raises:
            ValueError: If PDF is corrupted or cannot be processed.
        """
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Failed to open PDF: {e}")

        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "page_number": page_num + 1,
                    "text": text,
                })
        doc.close()

        if not pages:
            raise ValueError("PDF contains no extractable text.")

        return pages

    @staticmethod
    def extract_text_from_plain(content: str) -> list[dict]:
        """Handle plain text input as a single page.

        Args:
            content: Raw text string.

        Returns:
            List with single dict containing page 1 text.
        """
        if not content.strip():
            raise ValueError("Text content is empty.")
        return [{"page_number": 1, "text": content.strip()}]

    @staticmethod
    def generate_document_id() -> str:
        """Generate a unique document ID."""
        return str(uuid.uuid4())

"""Claim Extraction Agent — Breaks input text into atomic factual claims."""

import os
import json
import re
import uuid
from typing import Optional


class ClaimExtractionAgent:
    """Extracts atomic factual claims from user input text.

    Each claim represents a single, independently verifiable fact.
    Uses LLM for intelligent extraction, with a fallback sentence-based approach.
    """

    def __init__(self, llm_client=None):
        """Initialize with optional LLM client.

        Args:
            llm_client: Optional LLM client for intelligent extraction.
        """
        self.llm_client = llm_client

    def run(self, text: str, input_type: str = "answer") -> list[dict]:
        """Extract claims from input text.

        Args:
            text: Input text to decompose.
            input_type: Type of input (answer, explanation, summary).

        Returns:
            List of claim dicts with claim_id and claim_text.
        """
        if not text or not text.strip():
            return []

        if self.llm_client:
            try:
                return self._extract_with_llm(text, input_type)
            except Exception:
                pass

        return self._extract_with_rules(text)

    def _extract_with_llm(self, text: str, input_type: str) -> list[dict]:
        """Extract claims using LLM.

        Args:
            text: Input text.
            input_type: Type of input.

        Returns:
            List of claim dicts.
        """
        prompt = f"""Break the following {input_type} into atomic factual claims. 
Each claim must:
- Represent a single fact
- Be independently verifiable
- Preserve the original meaning

Return ONLY a JSON array of strings, each being one claim.

Text: {text}

Claims:"""

        response = self.llm_client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )

        content = response.choices[0].message.content.strip()
        # Parse JSON from response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            claims_list = json.loads(json_match.group())
            return [
                {"claim_id": str(uuid.uuid4()), "claim_text": claim.strip()}
                for claim in claims_list
                if claim.strip()
            ]
        
        raise ValueError("Could not parse LLM response")

    def _extract_with_rules(self, text: str) -> list[dict]:
        """Fallback: extract claims using sentence splitting.

        Args:
            text: Input text.

        Returns:
            List of claim dicts.
        """
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!])\s+', text.strip())

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out very short or non-factual sentences
            if len(sentence) < 10:
                continue
            if sentence.endswith("?"):
                continue

            claims.append({
                "claim_id": str(uuid.uuid4()),
                "claim_text": sentence,
            })

        return claims

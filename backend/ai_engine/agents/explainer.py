"""Explanation Agent — Generates human-readable explanations for verification results."""

import os
import json
import re
from typing import Optional


class ExplanationAgent:
    """Generates clear explanations for why a claim is supported, weak, or unsupported.

    References evidence content. Does NOT change the verification decision.
    """

    def __init__(self, llm_client=None):
        """Initialize with optional LLM client.

        Args:
            llm_client: Optional LLM client for generating explanations.
        """
        self.llm_client = llm_client

    def run(self, verification_results: list[dict]) -> list[dict]:
        """Generate explanations for each verified claim.

        Args:
            verification_results: List of verification result dicts.

        Returns:
            Same list with explanation field added.
        """
        for result in verification_results:
            if self.llm_client:
                try:
                    result["explanation"] = self._explain_with_llm(result)
                    continue
                except Exception:
                    pass

            result["explanation"] = self._explain_with_rules(result)

        return verification_results

    def _explain_with_llm(self, result: dict) -> str:
        """Generate explanation using LLM.

        Args:
            result: Verification result dict.

        Returns:
            Explanation string.
        """
        evidence_text = ""
        for i, e in enumerate(result.get("evidence", []), 1):
            evidence_text += f"\n  Evidence {i} (page {e.get('page_number', '?')}): {e.get('snippet', '')}"

        prompt = f"""Explain why this claim has the status '{result['status']}' with confidence {result['confidence_score']}.

Claim: {result['claim_text']}
Status: {result['status']}
Confidence: {result['confidence_score']}
Evidence: {evidence_text}

Write a concise explanation (2-3 sentences) that:
1. References the evidence
2. Explains why the claim has this status
3. Does NOT change the decision

Explanation:"""

        response = self.llm_client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )

        return response.choices[0].message.content.strip()

    def _explain_with_rules(self, result: dict) -> str:
        """Generate rule-based explanation.

        Args:
            result: Verification result dict.

        Returns:
            Explanation string.
        """
        status = result.get("status", "unsupported")
        confidence = result.get("confidence_score", 0)
        claim = result.get("claim_text", "")
        evidence = result.get("evidence", [])

        if status == "supported":
            if evidence:
                page_refs = ", ".join([f"page {e.get('page_number', '?')}" for e in evidence])
                return (
                    f"This claim is supported by evidence found in the uploaded documents "
                    f"({page_refs}). The retrieved content closely matches the assertion "
                    f"with a confidence score of {confidence}."
                )
            return f"This claim is marked as supported with confidence {confidence}."

        elif status == "weakly_supported":
            if evidence:
                page_refs = ", ".join([f"page {e.get('page_number', '?')}" for e in evidence])
                return (
                    f"This claim has partial support from the documents ({page_refs}). "
                    f"The evidence provides indirect or incomplete confirmation "
                    f"with a confidence score of {confidence}."
                )
            return f"This claim has weak support with confidence {confidence}."

        else:  # unsupported
            if evidence:
                return (
                    f"Despite retrieving potentially related content, the evidence does not "
                    f"sufficiently support this claim. The confidence score is {confidence}."
                )
            return (
                f"No supporting evidence was found in the uploaded documents for this claim. "
                f"The confidence score is {confidence}."
            )

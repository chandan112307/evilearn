"""Verification Agent — Evaluates claim correctness using evidence."""

from typing import Optional


class VerificationAgent:
    """Evaluates whether claims are supported by retrieved evidence.

    Assigns status and confidence score based ONLY on evidence.
    Does NOT use external knowledge or assumptions.
    """

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    MEDIUM_CONFIDENCE_THRESHOLD = 0.4

    def run(self, claims: list[dict], evidence_map: dict) -> list[dict]:
        """Verify each claim against its evidence.

        Args:
            claims: List of claim dicts.
            evidence_map: Dict mapping claim_id to evidence list.

        Returns:
            List of verification result dicts.
        """
        results = []

        for claim in claims:
            claim_id = claim["claim_id"]
            claim_text = claim["claim_text"]
            evidence_list = evidence_map.get(claim_id, [])

            verification = self._verify_claim(claim_text, evidence_list)
            verification["claim_id"] = claim_id
            verification["claim_text"] = claim_text
            verification["evidence"] = [
                {
                    "snippet": e.get("text_snippet", ""),
                    "page_number": e.get("page_number", 0),
                }
                for e in evidence_list[:3]  # Top 3 evidence pieces
            ]

            results.append(verification)

        return results

    def _verify_claim(self, claim_text: str, evidence_list: list[dict]) -> dict:
        """Verify a single claim against evidence.

        Args:
            claim_text: The claim to verify.
            evidence_list: Retrieved evidence chunks.

        Returns:
            Dict with status and confidence_score.
        """
        if not evidence_list:
            return {
                "status": "unsupported",
                "confidence_score": 0.1,
            }

        # Calculate aggregate relevance
        relevance_scores = [e.get("relevance_score", 0) for e in evidence_list]
        max_relevance = max(relevance_scores) if relevance_scores else 0
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

        # Check for keyword overlap as additional signal
        claim_words = set(claim_text.lower().split())
        best_overlap = 0
        for evidence in evidence_list:
            snippet = evidence.get("text_snippet", "")
            evidence_words = set(snippet.lower().split())
            if claim_words:
                overlap = len(claim_words & evidence_words) / len(claim_words)
                best_overlap = max(best_overlap, overlap)

        # Combined score
        combined_score = (max_relevance * 0.5) + (avg_relevance * 0.2) + (best_overlap * 0.3)

        if combined_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return {
                "status": "supported",
                "confidence_score": round(min(combined_score, 1.0), 2),
            }
        elif combined_score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return {
                "status": "weakly_supported",
                "confidence_score": round(combined_score, 2),
            }
        else:
            return {
                "status": "unsupported",
                "confidence_score": round(max(combined_score, 0.05), 2),
            }

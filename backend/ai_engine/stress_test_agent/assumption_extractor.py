"""Extract hidden assumptions from student reasoning."""

import os
import json
import re


def extract_assumptions(
    student_answer: str, claims: list[dict], llm_client=None
) -> list[str]:
    """Extract explicit and hidden assumptions from reasoning.

    Args:
        student_answer: The student's original answer.
        claims: Extracted claims from the answer.
        llm_client: Optional LLM client.

    Returns:
        List of assumption strings.
    """
    if not student_answer.strip():
        return []

    claim_texts = "\n".join(c.get("claim_text", "") for c in claims)

    # Try LLM
    if llm_client:
        try:
            prompt = (
                "You are a reasoning analyst. Identify ALL hidden and explicit "
                "assumptions in this student's reasoning.\n\n"
                "Focus on:\n"
                "- Implicit conditions not stated\n"
                "- Boundary assumptions (e.g., 'input is always positive')\n"
                "- Domain assumptions (e.g., 'function is continuous')\n"
                "- Logical assumptions (e.g., 'denominator is non-zero')\n\n"
                f"Student Answer:\n{student_answer}\n\n"
                f"Extracted Claims:\n{claim_texts}\n\n"
                "Return ONLY a JSON array of assumption strings.\n"
                "Assumptions:"
            )
            response = llm_client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                assumptions = json.loads(json_match.group())
                return [
                    a for a in assumptions if isinstance(a, str) and a.strip()
                ]
        except Exception:
            pass

    # Rule-based fallback
    assumptions = []
    text_lower = student_answer.lower()

    if any(w in text_lower for w in ["always", "all", "every", "any"]):
        assumptions.append(
            "Assumes the statement holds universally for all cases"
        )

    if any(
        w in text_lower
        for w in ["divide", "division", "denominator", "ratio", "/"]
    ):
        assumptions.append("Assumes denominator is non-zero")

    if any(
        w in text_lower
        for w in ["continuous", "differentiable", "smooth"]
    ):
        assumptions.append("Assumes function continuity or differentiability")

    if any(
        w in text_lower for w in ["positive", "greater than zero", "> 0"]
    ):
        assumptions.append("Assumes input values are positive")

    if any(
        w in text_lower
        for w in ["sorted", "ordered", "ascending", "descending"]
    ):
        assumptions.append("Assumes input is sorted or ordered")

    if any(w in text_lower for w in ["exists", "there is", "there are"]):
        assumptions.append(
            "Assumes existence of certain elements or solutions"
        )

    if not assumptions:
        assumptions.append("Input values are within expected domain")
        assumptions.append("Standard mathematical/logical rules apply")

    return assumptions

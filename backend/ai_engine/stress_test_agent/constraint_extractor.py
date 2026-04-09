"""Extract explicit and implicit constraints from problem and answer."""

import os
import json
import re


def extract_constraints(
    problem: str, student_answer: str, llm_client=None
) -> list[str]:
    """Extract constraints from problem statement and student answer.

    Args:
        problem: The problem statement (may be empty).
        student_answer: The student's answer.
        llm_client: Optional LLM client.

    Returns:
        List of constraint strings.
    """
    combined = f"{problem}\n{student_answer}".strip()
    if not combined:
        return []

    # Try LLM
    if llm_client:
        try:
            prompt = (
                "Extract ALL explicit and implicit constraints from this "
                "problem and answer.\n\n"
                "Look for:\n"
                "- Domain restrictions (e.g., 'x ≠ 0', 'x > 0')\n"
                "- Input constraints (e.g., 'input must be sorted')\n"
                "- Type constraints (e.g., 'values are integers')\n"
                "- Range constraints (e.g., 'between 0 and 100')\n"
                "- Boundary conditions\n\n"
            )
            if problem:
                prompt += f"Problem:\n{problem}\n\n"
            prompt += (
                f"Student Answer:\n{student_answer}\n\n"
                "Return ONLY a JSON array of constraint strings.\n"
                "Constraints:"
            )
            response = llm_client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                constraints = json.loads(json_match.group())
                return [
                    c for c in constraints if isinstance(c, str) and c.strip()
                ]
        except Exception:
            pass

    # Rule-based fallback
    constraints = []
    text = combined.lower()

    # Mathematical constraint patterns
    patterns = [
        (r'[a-z]\s*[≠!=]+\s*\d+', None),
        (r'[a-z]\s*[>≥]+\s*\d+', None),
        (r'[a-z]\s*[<≤]+\s*\d+', None),
    ]
    for pattern, _ in patterns:
        matches = re.findall(pattern, combined)
        for m in matches:
            constraints.append(m.strip())

    if "integer" in text or "whole number" in text:
        constraints.append("Values must be integers")
    if "positive" in text:
        constraints.append("Values must be positive")
    if "non-negative" in text or "nonnegative" in text:
        constraints.append("Values must be non-negative")
    if "sorted" in text:
        constraints.append("Input must be sorted")
    if "non-empty" in text or "not empty" in text:
        constraints.append("Input must be non-empty")
    if "finite" in text:
        constraints.append("Values must be finite")

    if not constraints:
        constraints.append("No explicit constraints stated")

    return constraints

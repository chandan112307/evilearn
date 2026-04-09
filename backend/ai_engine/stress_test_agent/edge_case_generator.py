"""Generate edge cases for stress testing."""

import os
import json
import re


def generate_edge_cases(
    concepts: list[str], constraints: list[str], llm_client=None
) -> list[str]:
    """Generate edge cases for stress testing reasoning.

    Hybrid approach: rule-based first, LLM fallback for domain-specific.

    Args:
        concepts: Key concepts from the reasoning.
        constraints: Extracted constraints.
        llm_client: Optional LLM client.

    Returns:
        List of edge case description strings.
    """
    edge_cases: list[str] = []

    # Rule-based edge case generation
    math_concepts = {
        "number", "value", "integer", "variable", "input", "output",
        "result", "sum", "product", "difference", "ratio", "root",
        "derivative", "integral", "function", "equation", "formula",
        "coefficient", "exponent", "polynomial", "fraction",
    }

    has_math = any(c.lower() in math_concepts for c in concepts)

    if has_math:
        edge_cases.extend(
            [
                "x = 0 (zero value)",
                "x < 0 (negative value)",
                "x is very large (approaching infinity)",
                "x is very small (approaching negative infinity)",
                "x = 1 (unit value)",
                "x = -1 (negative unit value)",
            ]
        )

    collection_concepts = {
        "list", "array", "string", "set", "sequence", "collection",
        "data", "elements", "items", "matrix", "vector",
    }
    has_collection = any(c.lower() in collection_concepts for c in concepts)

    if has_collection:
        edge_cases.extend(
            [
                "Empty input (no elements)",
                "Single element input",
                "Very large input size",
                "Duplicate elements in input",
            ]
        )

    # Constraint-based edge cases
    for constraint in constraints:
        c_lower = constraint.lower()
        if "positive" in c_lower:
            edge_cases.append(
                "Value is zero (boundary of positive constraint)"
            )
            edge_cases.append(
                "Value is negative (violates positive constraint)"
            )
        if "non-zero" in c_lower or "!= 0" in constraint or "≠ 0" in constraint:
            edge_cases.append(
                "Value equals zero (violates non-zero constraint)"
            )
        if "sorted" in c_lower:
            edge_cases.append(
                "Input is unsorted (violates sorted constraint)"
            )
            edge_cases.append("Input is reverse sorted")
        if "integer" in c_lower:
            edge_cases.append("Value is a non-integer (e.g., 2.5)")

    # Generic edge cases if none generated
    if not edge_cases:
        edge_cases = [
            "Zero/null input",
            "Negative values",
            "Extremely large values",
            "Boundary conditions",
            "Empty or minimal input",
        ]

    # LLM enhancement for domain-specific cases
    if llm_client:
        try:
            prompt = (
                "Generate additional edge cases for stress-testing reasoning "
                "about these concepts and constraints.\n\n"
                f"Concepts: {json.dumps(concepts[:10])}\n"
                f"Constraints: {json.dumps(constraints[:5])}\n"
                f"Already identified: {json.dumps(edge_cases[:5])}\n\n"
                "Return ONLY a JSON array of 3-5 NEW edge case strings "
                "not already listed.\nEdge cases:"
            )
            response = llm_client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                extra = json.loads(json_match.group())
                for e in extra:
                    if (
                        isinstance(e, str)
                        and e.strip()
                        and e not in edge_cases
                    ):
                        edge_cases.append(e.strip())
        except Exception:
            pass

    return edge_cases[:15]

"""Generate adversarial scenarios for stress testing."""

import os
import json
import re


def generate_adversarial_scenarios(
    weaknesses: list[dict],
    assumptions: list[str],
    constraints: list[str],
    edge_cases: list[str],
    llm_client=None,
) -> list[dict]:
    """Generate adversarial scenarios from weaknesses, assumptions, and edges.

    Args:
        weaknesses: Detected reasoning weaknesses.
        assumptions: Extracted assumptions.
        constraints: Extracted constraints.
        edge_cases: Generated edge cases.
        llm_client: Optional LLM client.

    Returns:
        List of {"scenario": str, "violates": str} dicts.
    """
    scenarios: list[dict] = []

    # Scenarios from assumption violations
    for assumption in assumptions:
        scenarios.append(
            {
                "scenario": f"What if: {assumption} is FALSE",
                "violates": f"assumption: {assumption}",
            }
        )

    # Scenarios from edge cases
    for edge_case in edge_cases:
        violated = "general boundary condition"
        for constraint in constraints:
            if any(
                word in edge_case.lower()
                for word in constraint.lower().split()[:3]
            ):
                violated = f"constraint: {constraint}"
                break

        scenarios.append(
            {
                "scenario": edge_case,
                "violates": violated,
            }
        )

    # Scenarios from constraint flips
    for constraint in constraints:
        if constraint != "No explicit constraints stated":
            scenarios.append(
                {
                    "scenario": f"Constraint violated: {constraint}",
                    "violates": f"constraint: {constraint}",
                }
            )

    # LLM enhancement for more targeted scenarios
    if llm_client and weaknesses:
        try:
            prompt = (
                "Generate targeted adversarial scenarios that would break "
                "reasoning with these weaknesses.\n\n"
                f"Weaknesses: {json.dumps(weaknesses[:5])}\n"
                f"Assumptions: {json.dumps(assumptions[:5])}\n"
                f"Constraints: {json.dumps(constraints[:5])}\n\n"
                "Return ONLY a JSON array of objects with 'scenario' and "
                "'violates' fields. Generate 3-5 scenarios.\n"
                "Scenarios:"
            )
            response = llm_client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                for s in parsed:
                    if (
                        isinstance(s, dict)
                        and "scenario" in s
                        and "violates" in s
                    ):
                        scenarios.append(
                            {
                                "scenario": str(s["scenario"]),
                                "violates": str(s["violates"]),
                            }
                        )
        except Exception:
            pass

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict] = []
    for s in scenarios:
        key = s["scenario"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique[:15]

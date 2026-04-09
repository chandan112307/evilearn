"""Detect weaknesses in student reasoning."""

import os
import json
import re


def analyze_weaknesses(
    claims: list[dict],
    assumptions: list[str],
    constraints: list[str],
    verification_results: list[dict],
    llm_client=None,
) -> list[dict]:
    """Detect reasoning weaknesses.

    Detects: overgeneralization, missing conditions, shallow reasoning,
    logical gaps, unjustified steps.

    Args:
        claims: Extracted claims.
        assumptions: Extracted assumptions.
        constraints: Extracted constraints.
        verification_results: Verification output from pipeline.
        llm_client: Optional LLM client.

    Returns:
        List of {"type": str, "detail": str} dicts.
    """
    weaknesses: list[dict] = []

    claim_texts = [c.get("claim_text", "") for c in claims]
    combined_claims = "\n".join(claim_texts)

    # Try LLM analysis
    if llm_client:
        try:
            prompt = (
                "You are a reasoning weakness detector. Analyze these claims "
                "and identify weaknesses in the reasoning.\n\n"
                "Weakness types to detect:\n"
                "- overgeneralization: claim applies too broadly\n"
                "- missing_condition: important conditions not specified\n"
                "- shallow_reasoning: steps lack depth or justification\n"
                "- logical_gap: missing logical connections\n"
                "- unjustified_step: conclusion without sufficient support\n\n"
                f"Claims:\n{combined_claims}\n\n"
                f"Assumptions:\n{json.dumps(assumptions)}\n\n"
                f"Constraints:\n{json.dumps(constraints)}\n\n"
                "Return ONLY a JSON array of objects with 'type' and "
                "'detail' fields.\nWeaknesses:"
            )
            response = llm_client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                for w in parsed:
                    if (
                        isinstance(w, dict)
                        and "type" in w
                        and "detail" in w
                    ):
                        weaknesses.append(
                            {
                                "type": str(w["type"]),
                                "detail": str(w["detail"]),
                            }
                        )
                if weaknesses:
                    return weaknesses
        except Exception:
            pass

    # Rule-based fallback
    for claim_text in claim_texts:
        text_lower = claim_text.lower()

        if any(
            w in text_lower
            for w in ["always", "never", "all", "every", "none"]
        ):
            weaknesses.append(
                {
                    "type": "overgeneralization",
                    "detail": (
                        f"Claim uses absolute language: "
                        f"'{claim_text[:80]}...'"
                    ),
                }
            )

        if any(w in text_lower for w in ["if", "when", "given", "assuming"]):
            if not any(
                w in text_lower for w in ["otherwise", "else", "except"]
            ):
                weaknesses.append(
                    {
                        "type": "missing_condition",
                        "detail": (
                            "Claim states a condition but doesn't address "
                            f"alternatives: '{claim_text[:80]}...'"
                        ),
                    }
                )

    for vr in verification_results:
        if vr.get("status") == "unsupported":
            weaknesses.append(
                {
                    "type": "unjustified_step",
                    "detail": (
                        "Claim lacks supporting evidence: "
                        f"'{vr.get('claim_text', '')[:80]}...'"
                    ),
                }
            )
        elif vr.get("status") == "weakly_supported":
            weaknesses.append(
                {
                    "type": "shallow_reasoning",
                    "detail": (
                        "Claim has only weak support: "
                        f"'{vr.get('claim_text', '')[:80]}...'"
                    ),
                }
            )

    if len(claims) > 1:
        weaknesses.append(
            {
                "type": "logical_gap",
                "detail": (
                    "Multiple claims present but logical connections "
                    "between them may not be explicit"
                ),
            }
        )

    if not weaknesses:
        weaknesses.append(
            {
                "type": "shallow_reasoning",
                "detail": (
                    "Reasoning lacks sufficient depth for thorough evaluation"
                ),
            }
        )

    return weaknesses

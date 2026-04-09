"""Convert failures and weaknesses into adversarial questions."""

import os
import json
import re


def generate_adversarial_questions(
    failures: list[dict],
    weaknesses: list[dict],
    llm_client=None,
) -> list[str]:
    """Convert failures into targeted adversarial questions.

    Rules:
    - Each failure maps to one question
    - Max 5 questions
    - No answers, only prompts

    Args:
        failures: Failure analysis results.
        weaknesses: Detected weaknesses.
        llm_client: Optional LLM client.

    Returns:
        List of adversarial question strings (max 5).
    """
    failed = [f for f in failures if f.get("fails", False)]

    if not failed and not weaknesses:
        return [
            "Can you identify any conditions under which your "
            "reasoning might not hold?"
        ]

    # Try LLM-based question generation
    if llm_client and failed:
        try:
            prompt = (
                "Convert these reasoning failures into sharp, direct "
                "questions that challenge the student's reasoning.\n\n"
                "Rules:\n"
                "- One question per failure\n"
                "- Questions only, no answers\n"
                "- Questions must directly target the failure point\n"
                "- Maximum 5 questions\n\n"
                f"Failures:\n{json.dumps(failed[:5])}\n\n"
                "Return ONLY a JSON array of question strings.\n"
                "Questions:"
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
                questions = json.loads(json_match.group())
                result = [
                    q
                    for q in questions
                    if isinstance(q, str) and q.strip()
                ]
                if result:
                    return result[:5]
        except Exception:
            pass

    # Rule-based fallback
    questions: list[str] = []

    for f in failed[:5]:
        scenario = f.get("scenario", "")
        scenario_lower = scenario.lower()

        if "= 0" in scenario_lower or "zero" in scenario_lower:
            questions.append(
                "What happens to your reasoning when the value is zero?"
            )
        elif "negative" in scenario_lower or "< 0" in scenario_lower:
            questions.append(
                "Does your reasoning hold for negative values?"
            )
        elif "empty" in scenario_lower or "null" in scenario_lower:
            questions.append(
                "Have you considered the case of empty or null input?"
            )
        elif (
            "large" in scenario_lower
            or "infinity" in scenario_lower
            or "∞" in scenario_lower
        ):
            questions.append(
                "How does your reasoning behave with very large values?"
            )
        elif "false" in scenario_lower or "violated" in scenario_lower:
            questions.append(
                f"What if {scenario.replace('What if: ', '').replace(' is FALSE', ' does not hold')}?"
            )
        else:
            failure_point = f.get("failure_point", "")
            if failure_point:
                questions.append(
                    f"Can you verify that '{failure_point}' "
                    "holds under all conditions?"
                )
            else:
                questions.append(
                    f"How does your reasoning handle: {scenario}?"
                )

    # Add weakness-based questions
    for w in weaknesses[:2]:
        w_type = w.get("type", "")
        if w_type == "overgeneralization" and len(questions) < 5:
            questions.append(
                "Are there cases where your generalization "
                "doesn't apply?"
            )
        elif w_type == "missing_condition" and len(questions) < 5:
            questions.append(
                "What conditions or exceptions have you not addressed?"
            )
        elif w_type == "logical_gap" and len(questions) < 5:
            questions.append(
                "Can you explicitly state the logical connection "
                "between your claims?"
            )

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:5]

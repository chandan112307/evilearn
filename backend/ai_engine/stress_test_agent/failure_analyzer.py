"""Analyze failures by simulating reasoning under adversarial scenarios.

CRITICAL MODULE: Implements the mandatory evaluation loop.
For each scenario, simulates reasoning step-by-step.
"""

import os
import json
import re


def analyze_failures(
    student_answer: str,
    scenarios: list[dict],
    llm_client=None,
) -> list[dict]:
    """Evaluate student reasoning under each adversarial scenario.

    CRITICAL: Implements true evaluation loop.
    For each scenario, simulates reasoning step-by-step.

    Args:
        student_answer: Original student answer.
        scenarios: List of {"scenario": str, "violates": str}.
        llm_client: Optional LLM client.

    Returns:
        List of failure analysis results.
    """
    if not scenarios:
        return []

    results: list[dict] = []

    for scenario in scenarios:
        scenario_text = scenario.get("scenario", "")
        violates = scenario.get("violates", "")

        failure_result = _evaluate_scenario(
            student_answer, scenario_text, violates, llm_client
        )
        results.append(failure_result)

    return results


def _evaluate_scenario(
    student_answer: str,
    scenario: str,
    violates: str,
    llm_client=None,
) -> dict:
    """Evaluate student reasoning under a single scenario.

    Simulates reasoning step-by-step under the given scenario
    and detects failure points.
    """
    # Try LLM-based analysis
    if llm_client:
        try:
            prompt = (
                "You are a reasoning failure detector. Simulate the student's "
                "reasoning step-by-step under this scenario.\n\n"
                f"Student's Answer:\n{student_answer}\n\n"
                f"Adversarial Scenario: {scenario}\n"
                f"This scenario violates: {violates}\n\n"
                "Instructions:\n"
                "1. Re-trace the student's reasoning step by step\n"
                "2. Apply the adversarial scenario at each step\n"
                "3. Determine if and where reasoning breaks\n"
                "4. Provide causal explanation for failure\n\n"
                "Return ONLY a JSON object with these fields:\n"
                '- "fails": boolean (does reasoning fail?)\n'
                '- "failure_point": string (which step fails, empty '
                "if no failure)\n"
                '- "reason": string (causal explanation of why it '
                "fails)\n\nResult:"
            )
            response = llm_client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "scenario": scenario,
                    "fails": bool(parsed.get("fails", True)),
                    "failure_point": str(
                        parsed.get("failure_point", "")
                    ),
                    "reason": str(parsed.get("reason", "")),
                }
        except Exception:
            pass

    # Rule-based fallback
    return _rule_based_evaluation(student_answer, scenario, violates)


def _rule_based_evaluation(
    student_answer: str,
    scenario: str,
    violates: str,
) -> dict:
    """Rule-based failure evaluation when LLM is unavailable."""
    answer_lower = student_answer.lower()
    scenario_lower = scenario.lower()

    fails = False
    failure_point = ""
    reason = ""

    # Zero-related scenarios
    if any(z in scenario_lower for z in ["= 0", "zero", "null", "empty"]):
        if any(
            w in answer_lower
            for w in ["divide", "division", "/", "denominator"]
        ):
            fails = True
            failure_point = "division step"
            reason = "Division by zero when input is zero"
        elif any(
            w in answer_lower for w in ["positive", "> 0", "greater than"]
        ):
            fails = True
            failure_point = "positivity assumption"
            reason = "Zero value violates the positive input assumption"
        else:
            fails = True
            failure_point = "boundary handling"
            reason = f"Reasoning does not account for: {scenario}"

    # Negative value scenarios
    elif any(
        n in scenario_lower
        for n in ["negative", "< 0", "less than zero"]
    ):
        if any(
            w in answer_lower
            for w in ["sqrt", "square root", "log", "logarithm"]
        ):
            fails = True
            failure_point = "domain restriction"
            reason = (
                "Negative values are outside the domain of "
                "sqrt/log operations"
            )
        elif "positive" in answer_lower:
            fails = True
            failure_point = "sign assumption"
            reason = (
                "Negative values violate the positive input assumption"
            )
        else:
            fails = True
            failure_point = "sign handling"
            reason = f"Reasoning may not handle: {scenario}"

    # Assumption violation scenarios
    elif "false" in scenario_lower or "violated" in scenario_lower:
        fails = True
        failure_point = "assumption dependency"
        reason = (
            f"Reasoning depends on violated condition: {violates}"
        )

    # Default: assume scenario causes failure for stress testing
    else:
        fails = True
        failure_point = "general reasoning step"
        reason = f"Reasoning not verified under scenario: {scenario}"

    return {
        "scenario": scenario,
        "fails": fails,
        "failure_point": failure_point,
        "reason": reason,
    }

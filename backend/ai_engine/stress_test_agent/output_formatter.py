"""Format stress test output into structured response."""


def format_output(
    failure_results: list[dict],
    weaknesses: list[dict],
    robustness: dict,
    questions: list[str],
) -> dict:
    """Format raw outputs into final response structure.

    Returns:
        Dict with stress_test_results, weakness_summary,
        robustness_summary, adversarial_questions.
    """
    # 1. Stress test results — natural language failure statements
    stress_test_results: list[str] = []
    for f in failure_results:
        scenario = f.get("scenario", "Unknown scenario")
        fails = f.get("fails", False)
        reason = f.get("reason", "")
        failure_point = f.get("failure_point", "")

        if fails:
            stmt = f"FAILS when: {scenario}"
            if failure_point:
                stmt += f" (at: {failure_point})"
            if reason:
                stmt += f" — {reason}"
        else:
            stmt = f"PASSES when: {scenario}"

        stress_test_results.append(stmt)

    # 2. Weakness summary
    weakness_summary = [
        {"type": w.get("type", ""), "detail": w.get("detail", "")}
        for w in weaknesses
    ]

    # 3. Robustness summary
    robustness_summary = robustness

    # 4. Adversarial questions
    adversarial_questions = questions

    return {
        "stress_test_results": stress_test_results,
        "weakness_summary": weakness_summary,
        "robustness_summary": robustness_summary,
        "adversarial_questions": adversarial_questions,
    }

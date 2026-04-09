"""Evaluate robustness of reasoning based on failure analysis."""


def evaluate_robustness(failure_results: list[dict]) -> dict:
    """Compute robustness score from failure results.

    score = passed_scenarios / total_scenarios

    Args:
        failure_results: List of failure analysis results.

    Returns:
        Dict with robustness_score, summary, and level.
    """
    if not failure_results:
        return {
            "robustness_score": 1.0,
            "summary": "No scenarios to evaluate",
            "level": "unknown",
        }

    total = len(failure_results)
    failed = sum(1 for r in failure_results if r.get("fails", False))
    passed = total - failed

    score = round(passed / total, 2) if total > 0 else 0.0

    # Determine level
    if score >= 0.8:
        level = "high"
    elif score >= 0.5:
        level = "medium"
    else:
        level = "low"

    summary = (
        f"Reasoning survives {passed} out of {total} scenarios "
        f"(score: {score})"
    )
    if failed > 0:
        summary += (
            f". Fails in {failed} scenario{'s' if failed > 1 else ''}"
        )

    return {
        "robustness_score": score,
        "summary": summary,
        "level": level,
    }

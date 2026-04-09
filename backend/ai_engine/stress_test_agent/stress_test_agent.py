"""Stress Test Agent — orchestrates all stress testing modules.

Main entry point for the Knowledge Stress-Test Engine.
Runs all modules in sequence and returns structured output.

This is NOT an answer generation system.
This is NOT a chatbot feature.
This is a reasoning stress-testing system.
"""

from .concept_extractor import extract_concepts
from .assumption_extractor import extract_assumptions
from .constraint_extractor import extract_constraints
from .weakness_analyzer import analyze_weaknesses
from .edge_case_generator import generate_edge_cases
from .adversarial_engine import generate_adversarial_scenarios
from .failure_analyzer import analyze_failures
from .robustness_evaluator import evaluate_robustness
from .adversarial_question_agent import generate_adversarial_questions
from .output_formatter import format_output


def run_stress_test(
    student_answer: str,
    claims: list[dict],
    verification_results: list[dict],
    problem: str = "",
    llm_client=None,
) -> dict:
    """Run complete stress test pipeline on student reasoning.

    Flow:
    1. concept_extractor
    2. assumption_extractor
    3. constraint_extractor
    4. weakness_analyzer
    5. edge_case_generator
    6. adversarial_engine
    7. failure_analyzer (evaluation loop)
    8. robustness_evaluator
    9. adversarial_question_agent
    10. output_formatter

    Args:
        student_answer: The student's original answer.
        claims: Extracted claims from the pipeline.
        verification_results: Verification output from the pipeline.
        problem: Optional problem statement.
        llm_client: Optional LLM client.

    Returns:
        Structured stress test output with:
        - stress_test_results: natural language failure statements
        - weakness_summary: pattern-level reasoning issues
        - robustness_summary: overall stability metrics
        - adversarial_questions: targeted challenge questions
    """
    # 1. Extract concepts
    concepts = extract_concepts(claims, llm_client)

    # 2. Extract assumptions
    assumptions = extract_assumptions(student_answer, claims, llm_client)

    # 3. Extract constraints
    constraints = extract_constraints(problem, student_answer, llm_client)

    # 4. Analyze weaknesses
    weaknesses = analyze_weaknesses(
        claims, assumptions, constraints, verification_results, llm_client
    )

    # 5. Generate edge cases
    edge_cases = generate_edge_cases(concepts, constraints, llm_client)

    # 6. Generate adversarial scenarios
    scenarios = generate_adversarial_scenarios(
        weaknesses, assumptions, constraints, edge_cases, llm_client
    )

    # 7. Analyze failures (evaluation loop — MANDATORY)
    failure_results = analyze_failures(
        student_answer, scenarios, llm_client
    )

    # 8. Evaluate robustness
    robustness = evaluate_robustness(failure_results)

    # 9. Generate adversarial questions
    questions = generate_adversarial_questions(
        failure_results, weaknesses, llm_client
    )

    # 10. Format output
    return format_output(failure_results, weaknesses, robustness, questions)

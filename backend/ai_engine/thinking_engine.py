"""Thinking Simulation Engine — LangGraph-based multi-agent reasoning simulator.

Simulates multiple reasoning approaches (beginner, intermediate, expert) for the
same problem, structurally analyzes them, and identifies gaps in the student's thinking.

This engine does NOT solve problems or check correctness. It only analyzes
reasoning structure, strategy differences, and abstraction levels.

Pipeline order (NON-NEGOTIABLE):
    START → cognitive_profile_generator → parallel_reasoning_generator
    → reasoning_structurer → strategy_tagger → comparative_analyzer
    → (if student exists) student_comparator → gap_generator → END

Tech stack: LangGraph StateGraph + LLM API (Groq/OpenAI) — nothing else.
"""

import os
import json
import re
import uuid
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# Shared State — every node reads from / writes to this TypedDict
# ---------------------------------------------------------------------------

class ThinkingState(TypedDict):
    """Shared state flowing through every node in the thinking simulation graph.

    All nodes MUST read/write from this shared state.
    """

    # Injected dependencies
    _llm_client: object

    # Input
    problem: str
    student_answer: str  # optional

    # Pipeline data (each field written by exactly one node)
    cognitive_profiles: list[dict]      # written by cognitive_profile_generator
    reasoning_paths: list[dict]         # written by parallel_reasoning_generator
    structured_graphs: list[dict]       # written by reasoning_structurer
    strategy_tags: list[dict]           # written by strategy_tagger
    comparison_results: dict            # written by comparative_analyzer
    student_comparison: dict            # written by student_comparator (conditional)
    gap_analysis: list[dict]            # written by gap_generator


# ---------------------------------------------------------------------------
# LLM Helper
# ---------------------------------------------------------------------------

def _llm_call(llm_client, prompt: str, max_tokens: int = 2048) -> str:
    """Make a single LLM call. Returns empty string on failure."""
    if not llm_client:
        return ""
    try:
        response = llm_client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


def _parse_json(text: str, fallback=None):
    """Extract and parse JSON from LLM response text."""
    if not text:
        return fallback
    # Try to find JSON object or array
    for pattern in [r'\{.*\}', r'\[.*\]']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                continue
    return fallback


# ---------------------------------------------------------------------------
# Node 1: Cognitive Profile Generator
# ---------------------------------------------------------------------------

def cognitive_profile_generator_node(state: ThinkingState) -> dict:
    """Generate exactly 3 distinct reasoning profiles: beginner, intermediate, expert.

    Reads: problem, _llm_client
    Writes: cognitive_profiles
    """
    problem = state["problem"]
    llm_client = state.get("_llm_client")

    profiles = []

    if llm_client:
        prompt = (
            "Generate exactly 3 distinct cognitive reasoning profiles for analyzing "
            "the following problem. Do NOT solve the problem.\n\n"
            f"Problem: {problem}\n\n"
            "Return a JSON array with exactly 3 objects, each having:\n"
            '- "level": one of "beginner", "intermediate", "expert"\n'
            '- "description": how this level approaches the problem\n'
            '- "characteristics": array of 3-4 reasoning traits\n\n'
            "Profile constraints:\n"
            "- Beginner: Direct formula usage, no transformations, linear/surface-level "
            "reasoning, no assumption checking\n"
            "- Intermediate: Step-by-step rule application, moderate decomposition, "
            "handles standard variations, limited abstraction\n"
            "- Expert: Problem reframing, transformation and reduction, high abstraction, "
            "optimal/efficient reasoning path\n\n"
            "JSON array:"
        )
        result = _parse_json(_llm_call(llm_client, prompt), None)
        if isinstance(result, list) and len(result) >= 3:
            for p in result[:3]:
                profiles.append({
                    "level": p.get("level", "unknown"),
                    "description": p.get("description", ""),
                    "characteristics": p.get("characteristics", []),
                })

    # Fallback: generate rule-based profiles
    if len(profiles) < 3:
        profiles = [
            {
                "level": "beginner",
                "description": "Applies formulas directly without transformation. "
                "Linear, surface-level reasoning with no assumption checking.",
                "characteristics": [
                    "Direct formula usage",
                    "No transformations",
                    "Linear reasoning",
                    "No assumption checking",
                ],
            },
            {
                "level": "intermediate",
                "description": "Applies rules step-by-step with moderate decomposition. "
                "Handles standard variations with limited abstraction.",
                "characteristics": [
                    "Step-by-step rule application",
                    "Moderate decomposition",
                    "Handles standard variations",
                    "Limited abstraction",
                ],
            },
            {
                "level": "expert",
                "description": "Reframes the problem, uses transformation and reduction. "
                "High abstraction with optimal reasoning paths.",
                "characteristics": [
                    "Problem reframing",
                    "Transformation and reduction",
                    "High abstraction",
                    "Optimal reasoning path",
                ],
            },
        ]

    return {"cognitive_profiles": profiles}


# ---------------------------------------------------------------------------
# Node 2: Parallel Reasoning Generator
# ---------------------------------------------------------------------------

def parallel_reasoning_generator_node(state: ThinkingState) -> dict:
    """Generate 3 independent reasoning paths, each following its profile constraints.

    Reads: problem, cognitive_profiles, _llm_client
    Writes: reasoning_paths
    """
    problem = state["problem"]
    profiles = state["cognitive_profiles"]
    llm_client = state.get("_llm_client")

    reasoning_paths = []

    for profile in profiles:
        level = profile["level"]
        characteristics = ", ".join(profile.get("characteristics", []))

        if llm_client:
            prompt = (
                f"Simulate how a {level}-level thinker would reason about this problem.\n"
                "Do NOT solve the problem. Only show the REASONING PROCESS.\n\n"
                f"Problem: {problem}\n\n"
                f"Cognitive profile: {profile['description']}\n"
                f"Constraints: {characteristics}\n\n"
                "Return a JSON object with:\n"
                '- "level": the cognitive level\n'
                '- "steps": array of step objects, each with:\n'
                '  - "step_id": unique identifier (e.g. "s1", "s2")\n'
                '  - "operation_type": type of reasoning operation\n'
                '  - "concept_used": concept or rule applied\n'
                '  - "input_value": what this step takes as input\n'
                '  - "output_value": what this step produces\n'
                '  - "reason": why this step was taken\n\n'
                "Generate 3-7 steps depending on the cognitive level.\n"
                "JSON object:"
            )
            result = _parse_json(_llm_call(llm_client, prompt), None)
            if isinstance(result, dict) and "steps" in result:
                path = {
                    "level": level,
                    "steps": [],
                }
                for s in result.get("steps", []):
                    path["steps"].append({
                        "step_id": s.get("step_id", str(uuid.uuid4())[:8]),
                        "operation_type": s.get("operation_type", "unknown"),
                        "concept_used": s.get("concept_used", ""),
                        "input_value": s.get("input_value", ""),
                        "output_value": s.get("output_value", ""),
                        "reason": s.get("reason", ""),
                    })
                reasoning_paths.append(path)
                continue

        # Fallback: rule-based reasoning path
        if level == "beginner":
            steps = [
                {
                    "step_id": "b1",
                    "operation_type": "identify",
                    "concept_used": "problem recognition",
                    "input_value": problem[:100],
                    "output_value": "Identified problem type",
                    "reason": "Read the problem statement directly",
                },
                {
                    "step_id": "b2",
                    "operation_type": "recall",
                    "concept_used": "formula recall",
                    "input_value": "Problem type",
                    "output_value": "Retrieved standard formula",
                    "reason": "Recalled the most common formula for this type",
                },
                {
                    "step_id": "b3",
                    "operation_type": "substitute",
                    "concept_used": "direct substitution",
                    "input_value": "Values from problem",
                    "output_value": "Substituted values",
                    "reason": "Plugged values directly into formula",
                },
                {
                    "step_id": "b4",
                    "operation_type": "compute",
                    "concept_used": "arithmetic",
                    "input_value": "Substituted expression",
                    "output_value": "Numerical result",
                    "reason": "Performed basic computation",
                },
            ]
        elif level == "intermediate":
            steps = [
                {
                    "step_id": "i1",
                    "operation_type": "analyze",
                    "concept_used": "problem decomposition",
                    "input_value": problem[:100],
                    "output_value": "Identified sub-problems",
                    "reason": "Broke the problem into manageable parts",
                },
                {
                    "step_id": "i2",
                    "operation_type": "classify",
                    "concept_used": "pattern matching",
                    "input_value": "Sub-problems",
                    "output_value": "Matched to known patterns",
                    "reason": "Recognized standard problem patterns",
                },
                {
                    "step_id": "i3",
                    "operation_type": "apply_rule",
                    "concept_used": "rule-based reasoning",
                    "input_value": "Matched patterns",
                    "output_value": "Applied appropriate rules",
                    "reason": "Selected and applied relevant rules",
                },
                {
                    "step_id": "i4",
                    "operation_type": "verify",
                    "concept_used": "consistency check",
                    "input_value": "Intermediate results",
                    "output_value": "Verified intermediate steps",
                    "reason": "Checked consistency of intermediate results",
                },
                {
                    "step_id": "i5",
                    "operation_type": "synthesize",
                    "concept_used": "integration",
                    "input_value": "Verified sub-results",
                    "output_value": "Combined result",
                    "reason": "Combined sub-problem results",
                },
            ]
        else:  # expert
            steps = [
                {
                    "step_id": "e1",
                    "operation_type": "reframe",
                    "concept_used": "problem reframing",
                    "input_value": problem[:100],
                    "output_value": "Reframed problem representation",
                    "reason": "Transformed problem into a more tractable form",
                },
                {
                    "step_id": "e2",
                    "operation_type": "abstract",
                    "concept_used": "abstraction",
                    "input_value": "Reframed problem",
                    "output_value": "Abstract structure identified",
                    "reason": "Identified underlying abstract structure",
                },
                {
                    "step_id": "e3",
                    "operation_type": "transform",
                    "concept_used": "transformation",
                    "input_value": "Abstract structure",
                    "output_value": "Simplified representation",
                    "reason": "Applied transformation to simplify",
                },
                {
                    "step_id": "e4",
                    "operation_type": "optimize",
                    "concept_used": "optimization",
                    "input_value": "Simplified form",
                    "output_value": "Optimal path identified",
                    "reason": "Found the most efficient reasoning path",
                },
                {
                    "step_id": "e5",
                    "operation_type": "validate",
                    "concept_used": "boundary analysis",
                    "input_value": "Solution path",
                    "output_value": "Validated against edge cases",
                    "reason": "Checked assumptions and edge cases",
                },
                {
                    "step_id": "e6",
                    "operation_type": "generalize",
                    "concept_used": "generalization",
                    "input_value": "Specific solution",
                    "output_value": "Generalized approach",
                    "reason": "Extended reasoning to general case",
                },
            ]

        reasoning_paths.append({"level": level, "steps": steps})

    return {"reasoning_paths": reasoning_paths}


# ---------------------------------------------------------------------------
# Node 3: Reasoning Structurer
# ---------------------------------------------------------------------------

def reasoning_structurer_node(state: ThinkingState) -> dict:
    """Convert each reasoning path into a structured graph representation.

    Reads: reasoning_paths, _llm_client
    Writes: structured_graphs
    """
    reasoning_paths = state["reasoning_paths"]
    llm_client = state.get("_llm_client")

    structured_graphs = []

    for path in reasoning_paths:
        level = path["level"]
        steps = path.get("steps", [])

        # Determine abstraction level
        if level == "beginner":
            abstraction_level = "low"
        elif level == "intermediate":
            abstraction_level = "medium"
        else:
            abstraction_level = "high"

        # Collect strategy types from operation types
        operation_types = [s.get("operation_type", "") for s in steps]
        strategy_types = list(set(operation_types))

        # Determine if reasoning is linear or transformed
        transformation_ops = {"transform", "reframe", "abstract", "optimize", "reduce"}
        has_transformation = any(
            op in transformation_ops for op in operation_types
        )

        # Build structured representation
        structured_steps = []
        for step in steps:
            structured_steps.append({
                "step_id": step.get("step_id", ""),
                "operation_type": step.get("operation_type", ""),
                "concept_used": step.get("concept_used", ""),
                "input": step.get("input_value", ""),
                "output": step.get("output_value", ""),
                "reason": step.get("reason", ""),
            })

        # If LLM is available, enrich with decisions
        decisions = []
        if llm_client and steps:
            prompt = (
                f"Given these reasoning steps for a {level}-level thinker:\n"
                f"{json.dumps(steps, indent=2)}\n\n"
                "Identify key decision points in this reasoning. "
                "Return a JSON array of strings, each describing a decision made.\n"
                "JSON array:"
            )
            result = _parse_json(_llm_call(llm_client, prompt, 512), None)
            if isinstance(result, list):
                decisions = [str(d) for d in result[:5]]

        if not decisions:
            # Fallback decisions
            if level == "beginner":
                decisions = ["Selected standard formula", "Applied direct substitution"]
            elif level == "intermediate":
                decisions = ["Chose decomposition strategy", "Selected matching rules",
                             "Decided to verify intermediate results"]
            else:
                decisions = ["Chose to reframe problem", "Selected transformation approach",
                             "Decided to validate against edge cases"]

        graph = {
            "level": level,
            "steps": structured_steps,
            "decisions": decisions,
            "metadata": {
                "abstraction_level": abstraction_level,
                "strategy_types": strategy_types,
                "step_count": len(steps),
                "is_linear": not has_transformation,
            },
        }
        structured_graphs.append(graph)

    return {"structured_graphs": structured_graphs}


# ---------------------------------------------------------------------------
# Node 4: Strategy Tagger
# ---------------------------------------------------------------------------

def strategy_tagger_node(state: ThinkingState) -> dict:
    """Tag each reasoning path with strategy categories.

    Tags: direct_application, rule_based_reasoning, transformation,
          reduction, optimization

    Reads: structured_graphs, _llm_client
    Writes: strategy_tags
    """
    graphs = state["structured_graphs"]
    llm_client = state.get("_llm_client")

    strategy_tags = []

    # Strategy tag mapping based on operation types
    tag_map = {
        "identify": "direct_application",
        "recall": "direct_application",
        "substitute": "direct_application",
        "compute": "direct_application",
        "analyze": "rule_based_reasoning",
        "classify": "rule_based_reasoning",
        "apply_rule": "rule_based_reasoning",
        "verify": "rule_based_reasoning",
        "synthesize": "rule_based_reasoning",
        "reframe": "transformation",
        "abstract": "transformation",
        "transform": "transformation",
        "reduce": "reduction",
        "simplify": "reduction",
        "optimize": "optimization",
        "generalize": "optimization",
        "validate": "rule_based_reasoning",
    }

    for graph in graphs:
        level = graph["level"]
        steps = graph.get("steps", [])
        tags = set()

        for step in steps:
            op_type = step.get("operation_type", "").lower()
            if op_type in tag_map:
                tags.add(tag_map[op_type])
            else:
                # Try LLM for unknown operation types
                if llm_client and op_type:
                    prompt = (
                        f"Classify this reasoning operation into exactly one category:\n"
                        f"Operation: {op_type}\n\n"
                        "Categories: direct_application, rule_based_reasoning, "
                        "transformation, reduction, optimization\n\n"
                        "Return only the category name."
                    )
                    result = _llm_call(llm_client, prompt, 64).strip().lower()
                    valid_tags = {"direct_application", "rule_based_reasoning",
                                  "transformation", "reduction", "optimization"}
                    if result in valid_tags:
                        tags.add(result)

        # Ensure at least one tag
        if not tags:
            if level == "beginner":
                tags = {"direct_application"}
            elif level == "intermediate":
                tags = {"rule_based_reasoning"}
            else:
                tags = {"transformation", "optimization"}

        strategy_tags.append({
            "level": level,
            "tags": sorted(list(tags)),
        })

    return {"strategy_tags": strategy_tags}


# ---------------------------------------------------------------------------
# Node 5: Comparative Analyzer
# ---------------------------------------------------------------------------

def comparative_analyzer_node(state: ThinkingState) -> dict:
    """Compare reasoning paths across structural, strategy, and abstraction dimensions.

    Reads: structured_graphs, strategy_tags, _llm_client
    Writes: comparison_results
    """
    graphs = state["structured_graphs"]
    tags = state["strategy_tags"]
    llm_client = state.get("_llm_client")

    # --- Structural comparison ---
    structural = {}
    for graph in graphs:
        level = graph["level"]
        metadata = graph.get("metadata", {})
        structural[level] = {
            "step_count": metadata.get("step_count", 0),
            "is_linear": metadata.get("is_linear", True),
            "abstraction_level": metadata.get("abstraction_level", "low"),
        }

    # --- Strategy comparison ---
    strategy = {}
    all_tags = set()
    for tag_entry in tags:
        level = tag_entry["level"]
        level_tags = tag_entry.get("tags", [])
        strategy[level] = level_tags
        all_tags.update(level_tags)

    # Find missing strategies per level
    missing_strategies = {}
    for tag_entry in tags:
        level = tag_entry["level"]
        level_tags = set(tag_entry.get("tags", []))
        missing_strategies[level] = sorted(list(all_tags - level_tags))

    # --- Abstraction comparison ---
    abstraction = {}
    level_order = {"low": 1, "medium": 2, "high": 3}
    for graph in graphs:
        level = graph["level"]
        abs_level = graph.get("metadata", {}).get("abstraction_level", "low")
        abstraction[level] = {
            "abstraction_level": abs_level,
            "abstraction_score": level_order.get(abs_level, 1),
        }

    # LLM-enhanced comparison if available
    key_differences = []
    if llm_client:
        prompt = (
            "Compare these three reasoning approaches:\n"
            f"Structural: {json.dumps(structural)}\n"
            f"Strategies: {json.dumps(strategy)}\n"
            f"Abstraction: {json.dumps(abstraction)}\n\n"
            "Identify 3-5 key differences between beginner, intermediate, and expert "
            "reasoning. Focus on reasoning structure, NOT correctness.\n"
            "Return a JSON array of strings.\n"
            "JSON array:"
        )
        result = _parse_json(_llm_call(llm_client, prompt, 512), None)
        if isinstance(result, list):
            key_differences = [str(d) for d in result[:5]]

    if not key_differences:
        key_differences = [
            "Beginner uses linear, direct application while expert uses transformation",
            "Step count varies: beginner is shorter, expert adds validation steps",
            "Expert reasoning includes problem reframing absent in lower levels",
            "Intermediate adds verification that beginner skips",
        ]

    comparison_results = {
        "structural": structural,
        "strategy": {
            "per_level": strategy,
            "missing_strategies": missing_strategies,
        },
        "abstraction": abstraction,
        "key_differences": key_differences,
    }

    return {"comparison_results": comparison_results}


# ---------------------------------------------------------------------------
# Node 6: Student Comparator (CONDITIONAL — only if student_answer exists)
# ---------------------------------------------------------------------------

def student_comparator_node(state: ThinkingState) -> dict:
    """Compare student reasoning against beginner/intermediate/expert paths.

    ONLY runs if student_answer exists.

    Reads: student_answer, structured_graphs, strategy_tags, _llm_client
    Writes: student_comparison
    """
    student_answer = state.get("student_answer", "")
    graphs = state["structured_graphs"]
    tags = state["strategy_tags"]
    llm_client = state.get("_llm_client")

    comparison = {
        "student_level_match": "unknown",
        "missing_steps": [],
        "missing_strategies": [],
        "inefficiencies": [],
        "abstraction_gaps": [],
    }

    if llm_client and student_answer:
        # Build context of all reasoning levels
        levels_context = ""
        for graph in graphs:
            level = graph["level"]
            steps_summary = "; ".join(
                f"{s.get('operation_type', '')}: {s.get('concept_used', '')}"
                for s in graph.get("steps", [])
            )
            levels_context += f"\n{level}: {steps_summary}"

        tags_context = ""
        for tag_entry in tags:
            tags_context += f"\n{tag_entry['level']}: {', '.join(tag_entry.get('tags', []))}"

        prompt = (
            "Compare this student's reasoning against three cognitive levels.\n\n"
            f"Student reasoning: {student_answer}\n\n"
            f"Reasoning approaches by level:{levels_context}\n\n"
            f"Strategy tags by level:{tags_context}\n\n"
            "Return a JSON object with:\n"
            '- "student_level_match": which level the student most resembles '
            '("beginner", "intermediate", "expert")\n'
            '- "missing_steps": array of steps the student is missing (strings)\n'
            '- "missing_strategies": array of strategies the student does not use\n'
            '- "inefficiencies": array of inefficiencies in student reasoning\n'
            '- "abstraction_gaps": array of abstraction improvements needed\n\n'
            "Do NOT judge correctness. Only analyze reasoning structure.\n"
            "JSON object:"
        )
        result = _parse_json(_llm_call(llm_client, prompt, 1024), None)
        if isinstance(result, dict):
            comparison = {
                "student_level_match": result.get("student_level_match", "unknown"),
                "missing_steps": result.get("missing_steps", []),
                "missing_strategies": result.get("missing_strategies", []),
                "inefficiencies": result.get("inefficiencies", []),
                "abstraction_gaps": result.get("abstraction_gaps", []),
            }
    elif student_answer:
        # Rule-based fallback
        student_lower = student_answer.lower()

        # Detect level match
        has_transformation = any(
            kw in student_lower
            for kw in ["transform", "reframe", "abstract", "reduce", "simplify", "optimize"]
        )
        has_rules = any(
            kw in student_lower
            for kw in ["step", "rule", "apply", "decompose", "break down", "verify"]
        )

        if has_transformation:
            comparison["student_level_match"] = "expert"
        elif has_rules:
            comparison["student_level_match"] = "intermediate"
        else:
            comparison["student_level_match"] = "beginner"

        # Detect missing strategies
        all_strategies = set()
        student_strategies = set()
        for tag_entry in tags:
            all_strategies.update(tag_entry.get("tags", []))

        if "direct" in student_lower or "formula" in student_lower:
            student_strategies.add("direct_application")
        if has_rules:
            student_strategies.add("rule_based_reasoning")
        if has_transformation:
            student_strategies.add("transformation")
            student_strategies.add("optimization")

        comparison["missing_strategies"] = sorted(
            list(all_strategies - student_strategies)
        )

    return {"student_comparison": comparison}


# ---------------------------------------------------------------------------
# Conditional edge: check if student answer exists
# ---------------------------------------------------------------------------

def _check_student_answer(state: ThinkingState) -> str:
    """Route to student_comparator if student_answer exists, else skip."""
    if state.get("student_answer", "").strip():
        return "has_student"
    return "no_student"


# ---------------------------------------------------------------------------
# Node 7: Gap Generator
# ---------------------------------------------------------------------------

def gap_generator_node(state: ThinkingState) -> dict:
    """Generate strict insight outputs about thinking gaps.

    Reads: comparison_results, student_comparison, structured_graphs,
           strategy_tags, _llm_client
    Writes: gap_analysis
    """
    comparison = state["comparison_results"]
    student_comp = state.get("student_comparison", {})
    graphs = state["structured_graphs"]
    tags = state["strategy_tags"]
    llm_client = state.get("_llm_client")
    student_answer = state.get("student_answer", "").strip()

    gaps = []

    # Always generate comparative gaps between levels
    key_diffs = comparison.get("key_differences", [])
    for diff in key_diffs:
        gaps.append({
            "insight": diff,
            "severity": "info",
        })

    # If student answer exists, generate student-specific gaps
    if student_answer and student_comp:
        level_match = student_comp.get("student_level_match", "unknown")

        if level_match != "unknown":
            gaps.append({
                "insight": (
                    f"Your approach follows {level_match}-level reasoning: "
                    f"{'direct application' if level_match == 'beginner' else 'rule-based application' if level_match == 'intermediate' else 'transformation-based reasoning'}"
                ),
                "severity": "info" if level_match == "expert" else "warning",
            })

        # Missing steps
        for step in student_comp.get("missing_steps", []):
            gaps.append({
                "insight": f"Missing step: {step}",
                "severity": "warning",
            })

        # Missing strategies
        for strategy in student_comp.get("missing_strategies", []):
            strategy_readable = strategy.replace("_", " ")
            gaps.append({
                "insight": f"Your reasoning lacks {strategy_readable} strategy",
                "severity": "warning",
            })

        # Inefficiencies
        for ineff in student_comp.get("inefficiencies", []):
            gaps.append({
                "insight": f"Inefficiency: {ineff}",
                "severity": "warning",
            })

        # Abstraction gaps
        for gap in student_comp.get("abstraction_gaps", []):
            gaps.append({
                "insight": f"Abstraction gap: {gap}",
                "severity": "critical",
            })

        # Generate LLM-enhanced gap insights
        if llm_client:
            prompt = (
                "Based on this student comparison analysis, generate 3-5 specific "
                "thinking gap insights.\n\n"
                f"Student level match: {level_match}\n"
                f"Missing steps: {student_comp.get('missing_steps', [])}\n"
                f"Missing strategies: {student_comp.get('missing_strategies', [])}\n"
                f"Inefficiencies: {student_comp.get('inefficiencies', [])}\n\n"
                "Format each insight as a direct statement like:\n"
                '- "Intermediate reasoning introduces step X, which is missing"\n'
                '- "Expert simplifies using transformation Y"\n'
                '- "Your reasoning lacks abstraction at step Z"\n'
                '- "You are using a longer path than necessary"\n\n'
                "Return a JSON array of objects with 'insight' and 'severity' "
                '(one of: "info", "warning", "critical").\n'
                "JSON array:"
            )
            result = _parse_json(_llm_call(llm_client, prompt, 512), None)
            if isinstance(result, list):
                for item in result[:5]:
                    if isinstance(item, dict):
                        severity = item.get("severity", "info")
                        if severity not in {"info", "warning", "critical"}:
                            severity = "info"
                        gaps.append({
                            "insight": item.get("insight", ""),
                            "severity": severity,
                        })
    else:
        # No student answer — generate general gaps between levels
        # Structural gaps
        structural = comparison.get("structural", {})
        beginner_steps = structural.get("beginner", {}).get("step_count", 0)
        expert_steps = structural.get("expert", {}).get("step_count", 0)
        if beginner_steps > 0 and expert_steps > 0:
            gaps.append({
                "insight": (
                    f"Beginner uses {beginner_steps} steps while expert uses "
                    f"{expert_steps} steps — expert adds validation and generalization"
                ),
                "severity": "info",
            })

        # Strategy gaps
        strategy_data = comparison.get("strategy", {})
        missing = strategy_data.get("missing_strategies", {})
        for level, missing_tags in missing.items():
            if missing_tags:
                readable_tags = ", ".join(t.replace("_", " ") for t in missing_tags)
                gaps.append({
                    "insight": (
                        f"{level.capitalize()} reasoning is missing: {readable_tags}"
                    ),
                    "severity": "info",
                })

    return {"gap_analysis": gaps}


# ---------------------------------------------------------------------------
# Graph Construction — builds the LangGraph StateGraph
# ---------------------------------------------------------------------------

def build_thinking_simulation_graph():
    """Build and compile the Thinking Simulation LangGraph StateGraph.

    Returns a compiled graph with execution flow:
        START → cognitive_profile_generator → parallel_reasoning_generator
        → reasoning_structurer → strategy_tagger → comparative_analyzer
        → (if student exists) student_comparator → gap_generator → END

    All nodes are pure functions. No classes. No wrappers.
    """
    graph = StateGraph(ThinkingState)

    # Register pure-function nodes
    graph.add_node("cognitive_profile_generator", cognitive_profile_generator_node)
    graph.add_node("parallel_reasoning_generator", parallel_reasoning_generator_node)
    graph.add_node("reasoning_structurer", reasoning_structurer_node)
    graph.add_node("strategy_tagger", strategy_tagger_node)
    graph.add_node("comparative_analyzer", comparative_analyzer_node)
    graph.add_node("student_comparator", student_comparator_node)
    graph.add_node("gap_generator", gap_generator_node)

    # Strict sequential edges (NON-NEGOTIABLE order)
    graph.add_edge(START, "cognitive_profile_generator")
    graph.add_edge("cognitive_profile_generator", "parallel_reasoning_generator")
    graph.add_edge("parallel_reasoning_generator", "reasoning_structurer")
    graph.add_edge("reasoning_structurer", "strategy_tagger")
    graph.add_edge("strategy_tagger", "comparative_analyzer")

    # Conditional edge: student comparator only if student_answer exists
    graph.add_conditional_edges(
        "comparative_analyzer",
        _check_student_answer,
        {"has_student": "student_comparator", "no_student": "gap_generator"},
    )
    graph.add_edge("student_comparator", "gap_generator")
    graph.add_edge("gap_generator", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API — used by the backend to execute the thinking simulation
# ---------------------------------------------------------------------------

class ThinkingSimulationEngine:
    """Entry point that holds runtime dependencies and invokes the graph.

    This is NOT an agent class. It only:
    1. Stores reference to llm_client
    2. Injects it into shared state
    3. Calls graph.invoke()

    All reasoning logic lives in the graph nodes above.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.graph = build_thinking_simulation_graph()

    def simulate(self, problem: str, student_answer: str = "") -> dict:
        """Execute the thinking simulation pipeline.

        Args:
            problem: The problem or question to simulate reasoning for.
            student_answer: Optional student answer to compare against.

        Returns:
            Dict with cognitive profiles, reasoning paths, strategy tags,
            comparison results, student comparison, and gap analysis.

        Raises:
            ValueError: If problem is empty.
        """
        if not problem or not problem.strip():
            raise ValueError("Problem text is empty.")

        initial_state: ThinkingState = {
            "_llm_client": self.llm_client,
            "problem": problem.strip(),
            "student_answer": student_answer.strip() if student_answer else "",
            "cognitive_profiles": [],
            "reasoning_paths": [],
            "structured_graphs": [],
            "strategy_tags": [],
            "comparison_results": {},
            "student_comparison": {},
            "gap_analysis": [],
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "cognitive_profiles": final_state.get("cognitive_profiles", []),
            "reasoning_paths": final_state.get("reasoning_paths", []),
            "strategy_tags": final_state.get("strategy_tags", []),
            "comparison_results": final_state.get("comparison_results", {}),
            "student_comparison": final_state.get("student_comparison", {}),
            "gap_analysis": final_state.get("gap_analysis", []),
        }

"""Cognitive Load Optimizer — LangGraph-based real-time reasoning flow regulator.

This system sits between reasoning output and the user. It does NOT change
the correctness of explanations — it controls HOW they are presented.

Architecture (LangGraph cyclic StateGraph):
    START → explanation_analyzer → user_state_tracker → load_estimator
    → control_engine → granularity_controller → explanation_restructurer
    → feedback_loop_manager → (conditional: loop back or END)

All nodes are pure functions operating on shared CognitiveLoadState.
The graph is cyclic — the feedback loop manager decides whether to
re-optimize or finalize.
"""

import os
import re
import json
import uuid
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END

from ..schemas import (
    ExplanationStep,
    UserCognitiveState,
    CognitiveLoadMetrics,
    ControlAction,
)


# ---------------------------------------------------------------------------
# Shared State
# ---------------------------------------------------------------------------

class CognitiveLoadState(TypedDict):
    """Shared state for the Cognitive Load Optimizer graph."""

    # Injected
    _llm_client: object

    # Input
    raw_explanation: str
    user_id: str

    # Explanation analysis (written by explanation_analyzer)
    steps: list[dict]           # list of ExplanationStep dicts
    concept_transitions: list[str]
    abstraction_levels: list[str]

    # User state (written by user_state_tracker)
    user_state: dict            # UserCognitiveState dict

    # Load metrics (written by load_estimator)
    load_metrics: dict          # CognitiveLoadMetrics dict

    # Control decisions (written by control_engine)
    load_state: str             # overload / optimal / underload
    reasoning_mode: str         # fine-grained / medium / coarse
    control_actions: list[dict] # list of ControlAction dicts

    # Restructured output (written by granularity_controller + restructurer)
    adapted_steps: list[dict]   # list of ExplanationStep dicts

    # Feedback loop (written by feedback_loop_manager)
    iteration: int
    max_iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# In-memory user state store (persistent across requests within process)
# ---------------------------------------------------------------------------

_user_states: dict[str, dict] = {}


def _get_user_state(user_id: str) -> dict:
    """Retrieve or initialize user cognitive state."""
    if user_id not in _user_states:
        state = UserCognitiveState(user_id=user_id)
        _user_states[user_id] = state.model_dump()
    return _user_states[user_id].copy()


def _save_user_state(user_id: str, state: dict) -> None:
    """Persist updated user state."""
    _user_states[user_id] = state.copy()


# ---------------------------------------------------------------------------
# Node 1: Explanation Analyzer
# ---------------------------------------------------------------------------

def explanation_analyzer_node(state: CognitiveLoadState) -> dict:
    """Break explanation into steps, concept transitions, and abstraction levels.

    Reads: raw_explanation, _llm_client
    Writes: steps, concept_transitions, abstraction_levels
    """
    raw = state["raw_explanation"]
    llm_client = state.get("_llm_client")

    steps = []
    concept_transitions = []
    abstraction_levels = []

    if llm_client:
        try:
            prompt = (
                "Analyze the following explanation and break it into reasoning steps.\n"
                "For each step, identify:\n"
                "- content: the text of that step\n"
                "- concepts: key concepts introduced\n"
                "- abstraction_level: 'concrete', 'semi-abstract', or 'abstract'\n"
                "- depends_on: list of step_ids this step depends on\n\n"
                "Return ONLY a JSON array of objects with keys: "
                "step_id, content, concepts, abstraction_level, depends_on.\n"
                "Use step_id format: 's1', 's2', etc.\n\n"
                f"Explanation:\n{raw}\n\nSteps:"
            )
            response = llm_client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                for s in parsed:
                    if isinstance(s, dict) and s.get("content"):
                        abs_level = s.get("abstraction_level", "concrete")
                        if abs_level not in {"concrete", "semi-abstract", "abstract"}:
                            abs_level = "concrete"
                        step = ExplanationStep(
                            step_id=s.get("step_id", f"s{len(steps)+1}"),
                            content=s["content"],
                            concepts=s.get("concepts", []),
                            abstraction_level=abs_level,
                            depends_on=s.get("depends_on", []),
                        )
                        steps.append(step.model_dump())
        except Exception:
            pass

    # Fallback: sentence-based splitting
    if not steps:
        sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 5:
                continue
            step = ExplanationStep(
                step_id=f"s{i+1}",
                content=sent,
                concepts=[],
                abstraction_level="concrete",
                depends_on=[f"s{i}"] if i > 0 else [],
            )
            steps.append(step.model_dump())

    # Compute transitions and levels
    for i in range(1, len(steps)):
        prev_concepts = set(steps[i-1].get("concepts", []))
        curr_concepts = set(steps[i].get("concepts", []))
        new_concepts = curr_concepts - prev_concepts
        if new_concepts:
            concept_transitions.append(
                f"s{i} → s{i+1}: introduced {', '.join(new_concepts)}"
            )
        abstraction_levels.append(steps[i].get("abstraction_level", "concrete"))

    if steps:
        abstraction_levels.insert(0, steps[0].get("abstraction_level", "concrete"))

    return {
        "steps": steps,
        "concept_transitions": concept_transitions,
        "abstraction_levels": abstraction_levels,
    }


# ---------------------------------------------------------------------------
# Node 2: User State Tracker
# ---------------------------------------------------------------------------

def user_state_tracker_node(state: CognitiveLoadState) -> dict:
    """Load and return the current user cognitive state.

    Reads: user_id
    Writes: user_state
    """
    user_id = state.get("user_id", "default")
    user_state = _get_user_state(user_id)
    return {"user_state": user_state}


# ---------------------------------------------------------------------------
# Node 3: Load Estimator
# ---------------------------------------------------------------------------

def load_estimator_node(state: CognitiveLoadState) -> dict:
    """Compute cognitive load from explanation structure.

    Cognitive load is derived from:
    - step_density: number of steps / total content length (normalized)
    - concept_gap: average new concepts per transition
    - memory_demand: max concurrent dependencies

    Reads: steps, concept_transitions
    Writes: load_metrics
    """
    steps = state.get("steps", [])
    transitions = state.get("concept_transitions", [])

    if not steps:
        metrics = CognitiveLoadMetrics()
        return {"load_metrics": metrics.model_dump()}

    # Step density: steps per 100 words
    total_words = sum(len(s.get("content", "").split()) for s in steps)
    step_density = (len(steps) / max(total_words, 1)) * 100

    # Concept gap: average new concepts introduced per step
    total_new_concepts = 0
    for t in transitions:
        # Count concepts mentioned in transition string
        parts = t.split("introduced ")
        if len(parts) > 1:
            concepts = parts[1].split(", ")
            total_new_concepts += len(concepts)
    concept_gap = total_new_concepts / max(len(steps) - 1, 1) if len(steps) > 1 else 0

    # Memory demand: max number of dependencies any single step has
    max_deps = 0
    for s in steps:
        deps = len(s.get("depends_on", []))
        # Also count concepts that must be held
        concepts_count = len(s.get("concepts", []))
        max_deps = max(max_deps, deps + concepts_count)
    memory_demand = float(max_deps)

    # Composite load: weighted combination (0-10 scale)
    total_load = min(
        (step_density * 2.0) + (concept_gap * 2.5) + (memory_demand * 1.5),
        10.0
    )

    metrics = CognitiveLoadMetrics(
        step_density=round(step_density, 2),
        concept_gap=round(concept_gap, 2),
        memory_demand=round(memory_demand, 2),
        total_load=round(total_load, 2),
    )
    return {"load_metrics": metrics.model_dump()}


# ---------------------------------------------------------------------------
# Node 4: Control Engine
# ---------------------------------------------------------------------------

def control_engine_node(state: CognitiveLoadState) -> dict:
    """Compare load vs user capacity and decide adaptation strategy.

    Reads: load_metrics, user_state
    Writes: load_state, reasoning_mode, control_actions
    """
    load_metrics = state.get("load_metrics", {})
    user_state = state.get("user_state", {})

    total_load = load_metrics.get("total_load", 5.0)
    understanding = user_state.get("understanding_level", 0.5)
    stability = user_state.get("reasoning_stability", 0.5)

    # User capacity: higher understanding + stability = higher capacity
    user_capacity = (understanding * 5.0) + (stability * 5.0)  # 0-10 scale

    # Compare load vs capacity
    control_actions = []

    if total_load > user_capacity + 1.5:
        load_state = "overload"
        reasoning_mode = "fine-grained"
        control_actions.append(ControlAction(
            action="split_steps",
            reason=f"Reducing complexity: splitting steps (load={total_load:.1f} > capacity={user_capacity:.1f})"
        ).model_dump())
        if load_metrics.get("concept_gap", 0) > 2.0:
            control_actions.append(ControlAction(
                action="add_intermediate",
                reason="Adding intermediate reasoning to bridge concept gaps"
            ).model_dump())
        if load_metrics.get("memory_demand", 0) > 4.0:
            control_actions.append(ControlAction(
                action="reduce_abstraction",
                reason="Reducing abstraction to lower memory demand"
            ).model_dump())
    elif total_load < user_capacity - 2.0:
        load_state = "underload"
        reasoning_mode = "coarse"
        control_actions.append(ControlAction(
            action="merge_steps",
            reason=f"Increasing abstraction: skipping basics (load={total_load:.1f} < capacity={user_capacity:.1f})"
        ).model_dump())
        if load_metrics.get("step_density", 0) > 3.0:
            control_actions.append(ControlAction(
                action="compress_reasoning",
                reason="Compressing reasoning: merging obvious steps"
            ).model_dump())
    else:
        load_state = "optimal"
        reasoning_mode = "medium"
        if total_load > user_capacity:
            control_actions.append(ControlAction(
                action="add_checkpoints",
                reason="Borderline load: adding checkpoints for safety"
            ).model_dump())
        else:
            control_actions.append(ControlAction(
                action="maintain",
                reason="Load matches capacity — maintaining current structure"
            ).model_dump())

    return {
        "load_state": load_state,
        "reasoning_mode": reasoning_mode,
        "control_actions": control_actions,
    }


# ---------------------------------------------------------------------------
# Node 5: Granularity Controller
# ---------------------------------------------------------------------------

def granularity_controller_node(state: CognitiveLoadState) -> dict:
    """Adjust step size based on control decisions.

    If overload: split large steps into smaller ones
    If underload: merge consecutive simple steps
    If optimal: keep as-is, possibly add checkpoints

    Reads: steps, load_state, reasoning_mode, control_actions
    Writes: adapted_steps
    """
    steps = state.get("steps", [])
    load_state = state.get("load_state", "optimal")
    actions = state.get("control_actions", [])
    action_types = {a.get("action", "") for a in actions}

    if not steps:
        return {"adapted_steps": []}

    adapted = []

    if load_state == "overload":
        # Split steps: break steps with long content into sub-steps
        for s in steps:
            content = s.get("content", "")
            words = content.split()
            if len(words) > 25 and "split_steps" in action_types:
                # Split into two sub-steps
                mid = len(words) // 2
                # Find nearest sentence boundary
                split_idx = mid
                for j in range(mid, min(mid + 10, len(words))):
                    if j > 0 and words[j - 1].endswith(('.', '!', '?', ',', ';')):
                        split_idx = j
                        break

                part1 = " ".join(words[:split_idx])
                part2 = " ".join(words[split_idx:])

                concepts = s.get("concepts", [])
                step1 = ExplanationStep(
                    step_id=f"{s['step_id']}a",
                    content=part1,
                    concepts=concepts[:len(concepts) // 2 + 1],
                    abstraction_level="concrete",
                    depends_on=s.get("depends_on", []),
                )
                step2 = ExplanationStep(
                    step_id=f"{s['step_id']}b",
                    content=part2,
                    concepts=concepts[len(concepts) // 2 + 1:],
                    abstraction_level=s.get("abstraction_level", "concrete"),
                    depends_on=[f"{s['step_id']}a"],
                )
                adapted.append(step1.model_dump())
                adapted.append(step2.model_dump())
            else:
                # Reduce abstraction if needed
                abs_level = s.get("abstraction_level", "concrete")
                if "reduce_abstraction" in action_types and abs_level == "abstract":
                    abs_level = "semi-abstract"
                elif "reduce_abstraction" in action_types and abs_level == "semi-abstract":
                    abs_level = "concrete"
                adapted.append({
                    **s,
                    "abstraction_level": abs_level,
                })

    elif load_state == "underload":
        # Merge steps: combine consecutive short steps
        i = 0
        while i < len(steps):
            if (
                i + 1 < len(steps)
                and "merge_steps" in action_types
                and len(steps[i].get("content", "").split()) < 15
                and len(steps[i + 1].get("content", "").split()) < 15
            ):
                merged_content = (
                    steps[i].get("content", "") + " " +
                    steps[i + 1].get("content", "")
                )
                merged_concepts = list(set(
                    steps[i].get("concepts", []) +
                    steps[i + 1].get("concepts", [])
                ))
                # Use higher abstraction level
                abs_map = {"concrete": 0, "semi-abstract": 1, "abstract": 2}
                rev_map = {0: "concrete", 1: "semi-abstract", 2: "abstract"}
                abs1 = abs_map.get(steps[i].get("abstraction_level", "concrete"), 0)
                abs2 = abs_map.get(steps[i + 1].get("abstraction_level", "concrete"), 0)

                merged_step = ExplanationStep(
                    step_id=steps[i]["step_id"],
                    content=merged_content.strip(),
                    concepts=merged_concepts,
                    abstraction_level=rev_map.get(max(abs1, abs2), "concrete"),
                    depends_on=steps[i].get("depends_on", []),
                )
                adapted.append(merged_step.model_dump())
                i += 2
            else:
                adapted.append(steps[i])
                i += 1
    else:
        # Optimal: keep as-is, maybe add checkpoints
        for i, s in enumerate(steps):
            adapted.append(s)
            # Add checkpoint after every 3 steps if borderline
            if (
                "add_checkpoints" in action_types
                and (i + 1) % 3 == 0
                and i + 1 < len(steps)
            ):
                checkpoint = ExplanationStep(
                    step_id=f"checkpoint_{i+1}",
                    content=f"[Checkpoint: Verify understanding of steps up to this point]",
                    concepts=[],
                    abstraction_level="concrete",
                    depends_on=[s["step_id"]],
                )
                adapted.append(checkpoint.model_dump())

    return {"adapted_steps": adapted}


# ---------------------------------------------------------------------------
# Node 6: Explanation Restructurer
# ---------------------------------------------------------------------------

def explanation_restructurer_node(state: CognitiveLoadState) -> dict:
    """Apply structural changes to the adapted steps.

    Ensures step IDs are consistent, dependencies are valid,
    and the explanation flows correctly.

    Reads: adapted_steps, reasoning_mode
    Writes: adapted_steps (cleaned/validated)
    """
    adapted = state.get("adapted_steps", [])
    reasoning_mode = state.get("reasoning_mode", "medium")

    if not adapted:
        return {"adapted_steps": []}

    # Re-index steps for consistency
    valid_ids = {s.get("step_id", "") for s in adapted}

    cleaned = []
    for s in adapted:
        # Clean dependencies: remove references to non-existent steps
        deps = [d for d in s.get("depends_on", []) if d in valid_ids]
        cleaned_step = ExplanationStep(
            step_id=s.get("step_id", f"s{len(cleaned)+1}"),
            content=s.get("content", ""),
            concepts=s.get("concepts", []),
            abstraction_level=s.get("abstraction_level", "concrete"),
            depends_on=deps,
        )
        cleaned.append(cleaned_step.model_dump())

    return {"adapted_steps": cleaned}


# ---------------------------------------------------------------------------
# Node 7: Feedback Loop Manager
# ---------------------------------------------------------------------------

def feedback_loop_manager_node(state: CognitiveLoadState) -> dict:
    """Update user state and determine whether to loop.

    After adaptation:
    1. Update user state based on current interaction
    2. Decide if another iteration is needed (load still not optimal)
    3. Save user state for future interactions

    Reads: user_state, load_state, load_metrics, iteration, max_iterations
    Writes: user_state, iteration, converged
    """
    user_state = state.get("user_state", {})
    load_state = state.get("load_state", "optimal")
    load_metrics = state.get("load_metrics", {})
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    # Update user state based on this interaction
    interaction_count = user_state.get("interaction_count", 0) + 1

    if load_state == "overload":
        # Decrease understanding estimate, increase overload signals
        understanding = max(0.0, user_state.get("understanding_level", 0.5) - 0.05)
        overload_signals = user_state.get("overload_signals", 0) + 1
        stability = max(0.0, user_state.get("reasoning_stability", 0.5) - 0.05)
    elif load_state == "underload":
        # Increase understanding estimate
        understanding = min(1.0, user_state.get("understanding_level", 0.5) + 0.05)
        overload_signals = max(0, user_state.get("overload_signals", 0) - 1)
        stability = min(1.0, user_state.get("reasoning_stability", 0.5) + 0.03)
    else:
        understanding = user_state.get("understanding_level", 0.5)
        overload_signals = user_state.get("overload_signals", 0)
        stability = min(1.0, user_state.get("reasoning_stability", 0.5) + 0.02)

    # Learning speed: based on how quickly we reach optimal
    learning_speed = user_state.get("learning_speed", 0.5)
    if load_state == "optimal":
        learning_speed = min(1.0, learning_speed + 0.02)

    updated_state = UserCognitiveState(
        user_id=user_state.get("user_id", "default"),
        understanding_level=round(understanding, 3),
        reasoning_stability=round(stability, 3),
        learning_speed=round(learning_speed, 3),
        overload_signals=overload_signals,
        interaction_count=interaction_count,
    )
    updated_dict = updated_state.model_dump()

    # Save to persistent store
    _save_user_state(updated_dict["user_id"], updated_dict)

    # Decide convergence
    new_iteration = iteration + 1
    converged = (load_state == "optimal") or (new_iteration >= max_iterations)

    return {
        "user_state": updated_dict,
        "iteration": new_iteration,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Conditional edge: loop or end
# ---------------------------------------------------------------------------

def _should_loop(state: CognitiveLoadState) -> str:
    """Decide whether to re-optimize or finalize."""
    if state.get("converged", True):
        return "end"
    return "loop"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_cognitive_load_graph():
    """Build and compile the LangGraph StateGraph for cognitive load optimization.

    Returns a compiled graph with cyclic feedback loop:
        START → explanation_analyzer → user_state_tracker → load_estimator
        → control_engine → granularity_controller → explanation_restructurer
        → feedback_loop_manager → (loop back to load_estimator OR END)
    """
    graph = StateGraph(CognitiveLoadState)

    # Register nodes
    graph.add_node("explanation_analyzer", explanation_analyzer_node)
    graph.add_node("user_state_tracker", user_state_tracker_node)
    graph.add_node("load_estimator", load_estimator_node)
    graph.add_node("control_engine", control_engine_node)
    graph.add_node("granularity_controller", granularity_controller_node)
    graph.add_node("explanation_restructurer", explanation_restructurer_node)
    graph.add_node("feedback_loop_manager", feedback_loop_manager_node)

    # Edges
    graph.add_edge(START, "explanation_analyzer")
    graph.add_edge("explanation_analyzer", "user_state_tracker")
    graph.add_edge("user_state_tracker", "load_estimator")
    graph.add_edge("load_estimator", "control_engine")
    graph.add_edge("control_engine", "granularity_controller")
    graph.add_edge("granularity_controller", "explanation_restructurer")
    graph.add_edge("explanation_restructurer", "feedback_loop_manager")

    # Cyclic feedback: loop back to load_estimator or end
    graph.add_conditional_edges(
        "feedback_loop_manager",
        _should_loop,
        {"loop": "load_estimator", "end": END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CognitiveLoadOptimizer:
    """Entry point for cognitive load optimization.

    Holds runtime dependencies and invokes the LangGraph.
    All logic lives in the graph nodes above.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.graph = build_cognitive_load_graph()

    def optimize(self, explanation: str, user_id: str = "default") -> dict:
        """Optimize an explanation for cognitive load.

        Args:
            explanation: Raw explanation text.
            user_id: User identifier for state tracking.

        Returns:
            Dict with adapted_explanation, load_state, control_actions,
            user_state, load_metrics, reasoning_mode.
        """
        if not explanation or not explanation.strip():
            raise ValueError("Explanation text is empty.")

        initial_state: CognitiveLoadState = {
            "_llm_client": self.llm_client,
            "raw_explanation": explanation,
            "user_id": user_id,
            "steps": [],
            "concept_transitions": [],
            "abstraction_levels": [],
            "user_state": {},
            "load_metrics": {},
            "load_state": "optimal",
            "reasoning_mode": "medium",
            "control_actions": [],
            "adapted_steps": [],
            "iteration": 0,
            "max_iterations": 3,
            "converged": False,
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "adapted_explanation": final_state.get("adapted_steps", []),
            "load_state": final_state.get("load_state", "optimal"),
            "control_actions": final_state.get("control_actions", []),
            "user_state": final_state.get("user_state", {}),
            "load_metrics": final_state.get("load_metrics", {}),
            "reasoning_mode": final_state.get("reasoning_mode", "medium"),
        }

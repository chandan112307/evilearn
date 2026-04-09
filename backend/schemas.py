"""Pydantic schemas for API request/response validation and internal pipeline typing.

ALL data flowing through the system MUST be strictly typed.
No untyped dictionaries in core flow.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# --- Request Schemas ---

class ProcessInputRequest(BaseModel):
    """Request body for processing user input."""
    input_text: str = Field(..., min_length=1, description="User's answer, summary, or explanation to validate.")


class FeedbackRequest(BaseModel):
    """Request body for submitting user feedback."""
    claim_id: str = Field(..., description="ID of the claim.")
    session_id: str = Field(..., description="ID of the session.")
    decision: str = Field(..., pattern="^(accept|reject)$", description="User decision: accept or reject.")


class EditClaimRequest(BaseModel):
    """Request body for editing and re-validating a claim."""
    claim_id: str = Field(..., description="Original claim ID.")
    session_id: str = Field(..., description="Session ID.")
    new_claim_text: str = Field(..., min_length=1, description="Edited claim text.")


# --- Response Schemas ---

class EvidenceItem(BaseModel):
    """Evidence supporting or contradicting a claim."""
    snippet: str
    page_number: int


class ClaimResult(BaseModel):
    """Result for a single verified claim."""
    claim_id: str
    claim_text: str
    status: str
    confidence_score: float
    evidence: list[EvidenceItem] = []
    explanation: str = ""

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"supported", "weakly_supported", "unsupported"}
        if v not in allowed:
            raise ValueError(f"Invalid status '{v}'. Must be one of: {allowed}")
        return v

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {v}")
        return v


class ProcessInputResponse(BaseModel):
    """Response for input processing."""
    session_id: str
    input_type: str
    claims: list[ClaimResult] = []
    message: str = ""


class DocumentResponse(BaseModel):
    """Response for document operations."""
    document_id: str
    file_name: str
    status: str
    page_count: int = 0
    message: str = ""


class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    feedback_id: str
    message: str = "Feedback recorded successfully."


# --- Internal Pipeline Schemas (strict typing for pipeline data) ---

class ClaimItem(BaseModel):
    """A single atomic claim extracted from user input."""
    claim_id: str
    claim_text: str


class EvidenceChunk(BaseModel):
    """A single evidence chunk retrieved from the document store."""
    text_snippet: str
    page_number: int = 0
    relevance_score: float = 0.0
    document_id: str = ""


class VerificationResult(BaseModel):
    """Verification result for a single claim."""
    claim_id: str
    claim_text: str
    status: str
    confidence_score: float
    evidence: list[EvidenceItem] = []

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"supported", "weakly_supported", "unsupported"}
        if v not in allowed:
            raise ValueError(f"Invalid status '{v}'. Must be one of: {allowed}")
        return v

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {v}")
        return v


class FinalClaimResult(BaseModel):
    """Final pipeline output for a single claim, with explanation."""
    claim_id: str
    claim_text: str
    status: str
    confidence_score: float
    evidence: list[EvidenceItem] = []
    explanation: str = ""

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"supported", "weakly_supported", "unsupported"}
        if v not in allowed:
            raise ValueError(f"Invalid status '{v}'. Must be one of: {allowed}")
        return v

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {v}")
        return v


# --- History Schemas (fully typed) ---

class HistoryClaimItem(BaseModel):
    """A claim stored in history."""
    claim_id: str
    session_id: str
    claim_text: str


class HistoryFeedbackItem(BaseModel):
    """A feedback entry stored in history."""
    feedback_id: str
    claim_id: str
    session_id: str
    user_decision: str
    created_at: str


class HistorySession(BaseModel):
    """A session in history — fully typed, no loose dicts."""
    session_id: str
    input_text: str
    input_type: Optional[str] = None
    created_at: str
    claims: list[HistoryClaimItem] = []
    results: list[ClaimResult] = []
    feedback: list[HistoryFeedbackItem] = []


class HistoryResponse(BaseModel):
    """Response for history retrieval."""
    sessions: list[HistorySession] = []


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str = ""


# --- Stress Test Schemas ---

class EvaluateReasoningRequest(BaseModel):
    """Request body for reasoning stress test."""
    problem: str = Field(default="", description="Problem statement (optional).")
    student_answer: str = Field(..., min_length=1, description="Student's answer to stress-test.")
    confidence: int = Field(default=50, ge=0, le=100, description="Student's confidence level (0-100).")


class WeaknessItem(BaseModel):
    """A single reasoning weakness."""
    type: str
    detail: str


class RobustnessSummary(BaseModel):
    """Robustness evaluation summary."""
    robustness_score: float = Field(ge=0.0, le=1.0)
    summary: str
    level: str

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = {"low", "medium", "high", "unknown"}
        if v not in allowed:
            raise ValueError(f"Invalid level '{v}'. Must be one of: {allowed}")
        return v


class EvaluateReasoningResponse(BaseModel):
    """Response for reasoning stress test."""
    stress_test_results: list[str] = []
    weakness_summary: list[WeaknessItem] = []
    robustness_summary: RobustnessSummary
    adversarial_questions: list[str] = []


# --- Thinking Simulation Engine Schemas (Graph-Based Cognitive Reasoning) ---

class ThinkingSimulationRequest(BaseModel):
    """Request body for thinking simulation."""
    problem: str = Field(..., min_length=1, description="Problem or question to simulate reasoning for.")
    student_answer: str = Field(default="", description="Optional student answer/reasoning to compare.")


class CognitiveProfile(BaseModel):
    """A cognitive reasoning profile with strict constraint rules."""
    level: str
    description: str
    characteristics: list[str] = []
    allowed_operations: list[str] = []
    forbidden_operations: list[str] = []
    max_abstraction: str = "LOW"

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = {"beginner", "intermediate", "expert"}
        if v not in allowed:
            raise ValueError(f"Invalid level '{v}'. Must be one of: {allowed}")
        return v

    @field_validator("max_abstraction")
    @classmethod
    def validate_max_abstraction(cls, v: str) -> str:
        allowed = {"LOW", "MEDIUM", "HIGH"}
        if v not in allowed:
            raise ValueError(f"Invalid max_abstraction '{v}'. Must be one of: {allowed}")
        return v


class ReasoningNode(BaseModel):
    """A single node in a reasoning graph."""
    step_id: str
    operation_type: str
    concept_used: str
    input_value: str = ""
    output_value: str = ""
    reasoning: str = ""
    abstraction_level: str = "LOW"
    strategy_type: str = "direct_application"

    @field_validator("abstraction_level")
    @classmethod
    def validate_abstraction_level(cls, v: str) -> str:
        allowed = {"LOW", "MEDIUM", "HIGH"}
        if v not in allowed:
            raise ValueError(f"Invalid abstraction_level '{v}'. Must be one of: {allowed}")
        return v

    @field_validator("strategy_type")
    @classmethod
    def validate_strategy_type(cls, v: str) -> str:
        allowed = {"direct_application", "rule_based", "transformation", "reduction", "optimization"}
        if v not in allowed:
            raise ValueError(f"Invalid strategy_type '{v}'. Must be one of: {allowed}")
        return v


class ReasoningEdge(BaseModel):
    """An edge connecting two nodes in a reasoning graph."""
    from_step_id: str
    to_step_id: str
    relation_type: str = "derives"

    @field_validator("relation_type")
    @classmethod
    def validate_relation_type(cls, v: str) -> str:
        allowed = {"derives", "transforms", "simplifies"}
        if v not in allowed:
            raise ValueError(f"Invalid relation_type '{v}'. Must be one of: {allowed}")
        return v


class DecisionPoint(BaseModel):
    """A decision point in a reasoning graph."""
    decision_point: str
    alternatives_considered: list[str] = []
    chosen_path_reason: str = ""


class AbstractionMetrics(BaseModel):
    """Abstraction scoring for a reasoning path."""
    average_abstraction: float = Field(ge=0.0, le=3.0)
    max_abstraction: str = "LOW"
    abstraction_transitions: list[str] = []
    abstraction_flow: list[str] = []

    @field_validator("max_abstraction")
    @classmethod
    def validate_max_abstraction(cls, v: str) -> str:
        allowed = {"LOW", "MEDIUM", "HIGH"}
        if v not in allowed:
            raise ValueError(f"Invalid max_abstraction '{v}'. Must be one of: {allowed}")
        return v


class ReasoningGraph(BaseModel):
    """A complete reasoning graph for one cognitive level."""
    level: str
    nodes: list[ReasoningNode] = []
    edges: list[ReasoningEdge] = []
    decisions: list[DecisionPoint] = []
    abstraction_metrics: AbstractionMetrics = AbstractionMetrics(average_abstraction=1.0)
    metadata: dict = {}


class StrategyDistribution(BaseModel):
    """Strategy distribution percentages for a reasoning path."""
    level: str
    direct_application_pct: float = 0.0
    rule_based_pct: float = 0.0
    transformation_pct: float = 0.0
    reduction_pct: float = 0.0
    optimization_pct: float = 0.0
    strategies_used: list[str] = []


class StructuralComparison(BaseModel):
    """Structural comparison across reasoning graphs."""
    graph_shape: dict = {}
    strategy_distribution: dict = {}
    abstraction_flow: dict = {}
    key_differences: list[str] = []


class StudentGraph(BaseModel):
    """Student reasoning converted to the same graph structure."""
    student_level_match: str = "unknown"
    nodes: list[ReasoningNode] = []
    edges: list[ReasoningEdge] = []
    abstraction_metrics: AbstractionMetrics = AbstractionMetrics(average_abstraction=1.0)
    missing_nodes: list[str] = []
    missing_transformations: list[str] = []
    unnecessary_steps: list[str] = []
    abstraction_mismatches: list[str] = []
    strategy_distribution: dict = {}


class GapItem(BaseModel):
    """A single thinking gap insight derived from structural analysis."""
    insight: str
    severity: str = "info"
    source: str = "structural"

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        allowed = {"info", "warning", "critical"}
        if v not in allowed:
            raise ValueError(f"Invalid severity '{v}'. Must be one of: {allowed}")
        return v

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        allowed = {"structural", "strategy", "abstraction", "comparison"}
        if v not in allowed:
            raise ValueError(f"Invalid source '{v}'. Must be one of: {allowed}")
        return v


class ThinkingSimulationResponse(BaseModel):
    """Response for thinking simulation — graph-based cognitive reasoning."""
    cognitive_profiles: list[CognitiveProfile] = []
    reasoning_graphs: list[ReasoningGraph] = []
    strategy_distributions: list[StrategyDistribution] = []
    structural_comparison: StructuralComparison = StructuralComparison()
    gap_analysis: list[GapItem] = []
    student_graph: StudentGraph = StudentGraph()
    validation_passed: bool = True
    validation_notes: list[str] = []


# --- Cognitive Load Optimizer Schemas ---

class CognitiveLoadRequest(BaseModel):
    """Request body for cognitive load optimization."""
    explanation: str = Field(..., min_length=1, description="Raw explanation text to optimize.")
    user_id: str = Field(default="default", description="User identifier for state tracking.")


class ExplanationStep(BaseModel):
    """A single step in a structured explanation."""
    step_id: str
    content: str
    concepts: list[str] = []
    abstraction_level: str = "concrete"
    depends_on: list[str] = []

    @field_validator("abstraction_level")
    @classmethod
    def validate_abstraction_level(cls, v: str) -> str:
        allowed = {"concrete", "semi-abstract", "abstract"}
        if v not in allowed:
            raise ValueError(f"Invalid abstraction_level '{v}'. Must be one of: {allowed}")
        return v


class UserCognitiveState(BaseModel):
    """Dynamic user cognitive state that evolves over interactions."""
    user_id: str = "default"
    understanding_level: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_stability: float = Field(default=0.5, ge=0.0, le=1.0)
    learning_speed: float = Field(default=0.5, ge=0.0, le=1.0)
    overload_signals: int = Field(default=0, ge=0)
    interaction_count: int = Field(default=0, ge=0)


class CognitiveLoadMetrics(BaseModel):
    """Measured cognitive load of an explanation."""
    step_density: float = Field(default=0.0, ge=0.0, description="Steps per unit of content.")
    concept_gap: float = Field(default=0.0, ge=0.0, description="Average conceptual jump between steps.")
    memory_demand: float = Field(default=0.0, ge=0.0, description="Elements to hold simultaneously.")
    total_load: float = Field(default=0.0, ge=0.0, description="Composite cognitive load score.")


class ControlAction(BaseModel):
    """A single adaptation action taken by the optimizer."""
    action: str
    reason: str


class CognitiveLoadResponse(BaseModel):
    """Response for cognitive load optimization."""
    adapted_explanation: list[ExplanationStep] = []
    load_state: str = "optimal"
    control_actions: list[ControlAction] = []
    user_state: UserCognitiveState = UserCognitiveState()
    load_metrics: CognitiveLoadMetrics = CognitiveLoadMetrics()
    reasoning_mode: str = "medium"

    @field_validator("load_state")
    @classmethod
    def validate_load_state(cls, v: str) -> str:
        allowed = {"overload", "optimal", "underload"}
        if v not in allowed:
            raise ValueError(f"Invalid load_state '{v}'. Must be one of: {allowed}")
        return v

    @field_validator("reasoning_mode")
    @classmethod
    def validate_reasoning_mode(cls, v: str) -> str:
        allowed = {"fine-grained", "medium", "coarse"}
        if v not in allowed:
            raise ValueError(f"Invalid reasoning_mode '{v}'. Must be one of: {allowed}")
        return v

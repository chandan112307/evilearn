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

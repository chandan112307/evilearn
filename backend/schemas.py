"""Pydantic schemas for API request and response validation."""

from pydantic import BaseModel, Field
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


class HistorySession(BaseModel):
    """A session in history."""
    session_id: str
    input_text: str
    input_type: Optional[str] = None
    created_at: str
    claims: list[dict] = []
    results: list[ClaimResult] = []
    feedback: list[dict] = []


class HistoryResponse(BaseModel):
    """Response for history retrieval."""
    sessions: list[HistorySession] = []


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str = ""

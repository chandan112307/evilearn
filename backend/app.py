"""EviLearn Backend — FastAPI Application.

Main API server implementing claim-based knowledge validation endpoints.
"""

import os
import uuid
import json
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .schemas import (
    ProcessInputRequest,
    ProcessInputResponse,
    FeedbackRequest,
    FeedbackResponse,
    EditClaimRequest,
    DocumentResponse,
    HistoryResponse,
    ClaimResult,
    EvidenceItem,
    ErrorResponse,
)
from .data_layer.document_processor import DocumentProcessor
from .data_layer.chunker import TextChunker
from .data_layer.vector_store import VectorStore
from .data_layer.database import Database
from .ai_engine.pipeline import ValidationPipeline


# --- Initialize Application ---

app = FastAPI(
    title="EviLearn API",
    description="Claim-based knowledge validation system API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Services ---

db = Database(db_path=settings.SQLITE_DB_PATH)
doc_processor = DocumentProcessor()
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
vector_store = VectorStore(
    persist_directory=settings.CHROMA_PERSIST_DIR,
    model_name=settings.EMBEDDING_MODEL,
)

# Initialize LLM client (optional)
llm_client = None
if settings.LLM_API_KEY:
    try:
        if settings.LLM_PROVIDER == "groq":
            from groq import Groq
            llm_client = Groq(api_key=settings.LLM_API_KEY)
        else:
            from openai import OpenAI
            llm_client = OpenAI(api_key=settings.LLM_API_KEY)
    except ImportError:
        pass

pipeline = ValidationPipeline(
    vector_store=vector_store,
    llm_client=llm_client,
)


# --- Health Check ---

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "EviLearn API", "version": "1.0.0"}


# --- Document Endpoints ---

@app.post("/upload-documents", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document (PDF or text file).

    Executes: Upload → Extract → Chunk → Embed → Store
    """
    # Validate file extension
    file_name = file.filename or "unknown"
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # Read file
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Check file size
    if len(file_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB}MB.",
        )

    # Generate document ID
    document_id = DocumentProcessor.generate_document_id()

    # Store document record
    db.insert_document(document_id=document_id, file_name=file_name)

    try:
        # Extract text
        if ext == ".pdf":
            pages = DocumentProcessor.extract_text_from_pdf(file_bytes)
        else:
            content = file_bytes.decode("utf-8", errors="ignore")
            pages = DocumentProcessor.extract_text_from_plain(content)

        # Chunk text
        chunks = chunker.chunk_pages(pages, document_id)

        # Store in vector database (ChromaDB generates embeddings automatically)
        chunk_texts = [c["chunk_text"] for c in chunks]
        vector_store.add_chunks(
            chunk_ids=[c["chunk_id"] for c in chunks],
            documents=chunk_texts,
            metadatas=[
                {"page_number": c["page_number"], "document_id": c["document_id"]}
                for c in chunks
            ],
        )

        # Store chunks in SQLite
        db.insert_chunks(chunks)

        # Update status
        db.update_document_status(document_id, "ready")

        return DocumentResponse(
            document_id=document_id,
            file_name=file_name,
            status="ready",
            page_count=len(pages),
            message=f"Document processed successfully. {len(chunks)} chunks created.",
        )

    except ValueError as e:
        db.update_document_status(document_id, "failed")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.update_document_status(document_id, "failed")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {e}")


@app.get("/documents")
def list_documents():
    """List all uploaded documents."""
    documents = db.get_documents()
    return {"documents": documents}


# --- Validation Endpoints ---

@app.post("/process-input", response_model=ProcessInputResponse)
def process_input(request: ProcessInputRequest):
    """Process user input through the validation pipeline.

    Executes: Input → Claims → Retrieval → Verification → Explanation → Output
    """
    # Check if documents exist
    documents = db.get_documents()
    ready_docs = [d for d in documents if d.get("status") == "ready"]
    if not ready_docs:
        raise HTTPException(
            status_code=400,
            detail="No knowledge base available. Please upload documents first.",
        )

    try:
        # Execute pipeline
        result = pipeline.execute(request.input_text)

        # Create session
        session_id = db.create_session(
            input_text=request.input_text,
            input_type=result.get("input_type", "answer"),
        )

        claims = result.get("claims", [])

        if not claims:
            return ProcessInputResponse(
                session_id=session_id,
                input_type=result.get("input_type", "answer"),
                claims=[],
                message=result.get("message", "No claims extracted."),
            )

        # Store claims
        db.insert_claims(
            session_id=session_id,
            claims=[{"claim_id": c["claim_id"], "claim_text": c["claim_text"]} for c in claims],
        )

        # Store results
        db.insert_results(session_id=session_id, results=claims)

        # Format response
        claim_results = []
        for c in claims:
            evidence_items = [
                EvidenceItem(snippet=e.get("snippet", ""), page_number=e.get("page_number", 0))
                for e in c.get("evidence", [])
            ]
            claim_results.append(
                ClaimResult(
                    claim_id=c["claim_id"],
                    claim_text=c["claim_text"],
                    status=c.get("status", "unsupported"),
                    confidence_score=c.get("confidence_score", 0.0),
                    evidence=evidence_items,
                    explanation=c.get("explanation", ""),
                )
            )

        return ProcessInputResponse(
            session_id=session_id,
            input_type=result.get("input_type", "answer"),
            claims=claim_results,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Results Endpoints ---

@app.get("/get-results/{session_id}")
def get_results(session_id: str):
    """Get validation results for a specific session."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    results = db.get_results_by_session(session_id)
    claims = db.get_claims_by_session(session_id)

    return {
        "session_id": session_id,
        "input_text": session.get("input_text", ""),
        "input_type": session.get("input_type", ""),
        "claims": results,
    }


# --- Feedback Endpoints ---

@app.post("/submit-feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest):
    """Submit user feedback (accept/reject) for a claim."""
    # Verify session exists
    session = db.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    feedback_id = db.insert_feedback(
        claim_id=request.claim_id,
        session_id=request.session_id,
        decision=request.decision,
    )

    return FeedbackResponse(
        feedback_id=feedback_id,
        message=f"Feedback '{request.decision}' recorded for claim {request.claim_id}.",
    )


@app.post("/edit-claim", response_model=ProcessInputResponse)
def edit_claim(request: EditClaimRequest):
    """Edit a claim and re-run the validation pipeline for it."""
    # Verify session exists
    session = db.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        # Re-run pipeline for the edited claim text
        result = pipeline.execute(request.new_claim_text)

        claims = result.get("claims", [])

        if claims:
            db.insert_claims(
                session_id=request.session_id,
                claims=[{"claim_id": c["claim_id"], "claim_text": c["claim_text"]} for c in claims],
            )
            db.insert_results(session_id=request.session_id, results=claims)

        claim_results = []
        for c in claims:
            evidence_items = [
                EvidenceItem(snippet=e.get("snippet", ""), page_number=e.get("page_number", 0))
                for e in c.get("evidence", [])
            ]
            claim_results.append(
                ClaimResult(
                    claim_id=c["claim_id"],
                    claim_text=c["claim_text"],
                    status=c.get("status", "unsupported"),
                    confidence_score=c.get("confidence_score", 0.0),
                    evidence=evidence_items,
                    explanation=c.get("explanation", ""),
                )
            )

        return ProcessInputResponse(
            session_id=request.session_id,
            input_type=result.get("input_type", "answer"),
            claims=claim_results,
        )

    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- History Endpoints ---

@app.get("/history")
def get_history():
    """Retrieve complete validation history."""
    history = db.get_history()
    return {"sessions": history}

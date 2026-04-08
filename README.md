# EviLearn — Claim-Based Knowledge Validation System

> **EviLearn is NOT a chatbot.** It is a structured, evidence-based reasoning system that validates user-submitted knowledge (answers, summaries, explanations) against uploaded reference documents. Every output is traceable to document evidence.

## Core Mechanism

```
Input → Claims → Evidence → Verification → Explanation
```

A user submits text. The system decomposes it into atomic claims, retrieves evidence from uploaded documents, verifies each claim against that evidence, and generates a human-readable explanation for every verdict. No claim is accepted or rejected without document-backed reasoning.

## System Architecture

EviLearn is organized into 4 layers, each with a strict single responsibility:

```mermaid
graph TB
    subgraph "Layer 1 — Frontend (React)"
        UI[React UI<br/>Vite + Tailwind CSS]
    end

    subgraph "Layer 2 — Backend API (FastAPI)"
        API[FastAPI Server<br/>Pydantic Validation]
    end

    subgraph "Layer 3 — AI Reasoning Engine"
        PL[Planner Agent]
        CE[Claim Extractor Agent]
        RA[Retrieval Agent]
        VA[Verification Agent]
        EA[Explanation Agent]
        PL --> CE --> RA --> VA --> EA
    end

    subgraph "Layer 4 — Data & Knowledge Layer"
        VEC[(ChromaDB<br/>Vector Store<br/>storage only)]
        SQL[(SQLite<br/>Relational DB)]
        DP[Document Processor<br/>PyMuPDF]
        EMB[EmbeddingService<br/>LLM API]
    end

    UI <-->|REST API| API
    API --> PL
    EA --> API
    RA <--> EMB
    EMB <--> VEC
    API <--> SQL
    API --> DP --> EMB --> VEC
```

| Layer | Responsibility | Technology |
|-------|---------------|------------|
| Frontend | Structured reasoning interface | React 19, Vite 8, Tailwind CSS 4 |
| Backend API | Request routing, validation, orchestration | FastAPI, Uvicorn, Pydantic |
| AI Engine | Deterministic multi-agent reasoning pipeline | LangGraph StateGraph |
| Data Layer | Document processing, embedding, storage, retrieval | PyMuPDF, EmbeddingService (LLM API), ChromaDB, SQLite |

## Data Flow Diagrams

### Document Upload Flow

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI
    participant DP as DocumentProcessor
    participant CH as TextChunker
    participant ES as EmbeddingService
    participant VS as ChromaDB
    participant DB as SQLite

    U->>FE: Upload PDF/TXT file
    FE->>API: POST /upload-documents (multipart)
    API->>API: Validate file type & size
    API->>DB: Insert document record (status: processing)
    API->>DP: Extract text with page mapping
    DP-->>API: List of {page_number, text}
    API->>CH: chunk_pages(pages, document_id)
    CH-->>API: List of chunks with IDs
    API->>ES: embed_texts(chunk_texts)
    Note over ES: Embeddings generated via LLM API
    ES-->>API: Pre-computed embedding vectors
    API->>VS: add_chunks(ids, documents, metadatas, embeddings)
    Note over VS: ChromaDB stores pre-computed<br/>embeddings — no auto-embedding
    API->>DB: Insert chunk records
    API->>DB: Update document status → ready
    API-->>FE: DocumentResponse
    FE-->>U: Success message + document list
```

### Validation Query Flow

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI
    participant PIP as LangGraph Pipeline
    participant VS as ChromaDB
    participant DB as SQLite

    U->>FE: Submit text for validation
    FE->>API: POST /process-input {input_text}
    API->>API: Check ready documents exist
    API->>PIP: pipeline.execute(input_text)

    Note over PIP: Stage 1 — Planner
    PIP->>PIP: Detect input type (answer/explanation/summary/question)

    Note over PIP: Stage 2 — Claim Extractor
    PIP->>PIP: Decompose into atomic claims (LLM or rules)

    Note over PIP: Stage 3 — Retriever
    PIP->>PIP: Generate query embedding via EmbeddingService
    PIP->>VS: query(query_embedding, top_k=5) per claim
    VS-->>PIP: Evidence chunks with relevance scores

    Note over PIP: Stage 4 — Verifier
    PIP->>PIP: Score claims → supported/weakly_supported/unsupported

    Note over PIP: Stage 5 — Explainer
    PIP->>PIP: Generate explanations (LLM or rules)

    PIP-->>API: Final results with claims, evidence, verdicts
    API->>API: Validate output via ClaimResult BEFORE storage
    API->>DB: Create session, store validated claims & results
    API-->>FE: ProcessInputResponse
    FE-->>U: Structured claim cards with evidence
```

### Feedback Flow

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI
    participant DB as SQLite

    U->>FE: Click Accept/Reject on claim
    FE->>API: POST /submit-feedback {claim_id, session_id, decision}
    API->>DB: Insert feedback record
    API-->>FE: FeedbackResponse
    FE-->>U: Visual confirmation

    U->>FE: Edit claim text
    FE->>API: POST /edit-claim {claim_id, session_id, new_claim_text}
    API->>API: Re-run full pipeline on edited text
    API->>DB: Store new claims & results
    API-->>FE: New ProcessInputResponse
    FE-->>U: Updated claim cards
```

## User Flow

1. **Upload Documents** — User uploads PDF or TXT files that form the knowledge base.
2. **Enter Text** — User submits an answer, summary, or explanation to validate.
3. **View Results** — System displays each extracted claim with a status badge, confidence score, evidence snippets, and explanation.
4. **Provide Feedback** — User can accept, reject, or edit individual claims.
5. **Review History** — User can browse past validation sessions with full results.

## API Endpoint Summary

| Method | Path | Description | Request | Response |
|--------|------|-------------|---------|----------|
| `GET` | `/` | Health check | — | `{status, service, version}` |
| `POST` | `/upload-documents` | Upload & process document | `multipart/form-data` (file) | `DocumentResponse` |
| `GET` | `/documents` | List all documents | — | `{documents: [...]}` |
| `POST` | `/process-input` | Validate text input | `{input_text: string}` | `ProcessInputResponse` |
| `GET` | `/get-results/{session_id}` | Get session results | — | `{session_id, input_text, input_type, claims}` |
| `POST` | `/submit-feedback` | Submit accept/reject | `{claim_id, session_id, decision}` | `FeedbackResponse` |
| `POST` | `/edit-claim` | Edit & re-validate claim | `{claim_id, session_id, new_claim_text}` | `ProcessInputResponse` |
| `GET` | `/history` | Get full history | — | `{sessions: [...]}` |

## Output Contract

Every verified claim returns this structure:

```json
{
  "claim_id": "uuid",
  "claim_text": "The atomic factual claim",
  "status": "supported | weakly_supported | unsupported",
  "confidence_score": 0.0 - 1.0,
  "evidence": [
    {
      "snippet": "Relevant text from document",
      "page_number": 1
    }
  ],
  "explanation": "Human-readable reasoning for the verdict"
}
```

### Status Definitions

| Status | Confidence Range | Meaning |
|--------|-----------------|---------|
| `supported` | ≥ 0.7 | Strong evidence match in documents |
| `weakly_supported` | 0.4 – 0.69 | Partial or indirect evidence found |
| `unsupported` | < 0.4 | No sufficient evidence in documents |

## Tech Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Frontend Framework | React | 19.x | UI components |
| Build Tool | Vite | 8.x | Dev server & bundling |
| CSS | Tailwind CSS | 4.x | Utility-first styling |
| Backend Framework | FastAPI | 0.104+ | REST API server |
| ASGI Server | Uvicorn | 0.24+ | Production server |
| Validation | Pydantic | 2.5+ | Request/response schemas |
| PDF Processing | PyMuPDF (fitz) | 1.23+ | Text extraction |
| Vector Database | ChromaDB | 0.4+ | Vector storage & similarity search (storage only) |
| Embedding | EmbeddingService | — | Embedding generation via LLM API (OpenAI/Groq) |
| Pipeline | LangGraph | 0.0.20+ | StateGraph orchestration |
| Relational DB | SQLite | Built-in | Session, claims, feedback storage |
| LLM Provider | Groq / OpenAI | — | Claim extraction & explanation |
| LLM Model | Llama 3 8B | — | Default model (configurable) |

## Setup & Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm 9+

### Backend

```bash
cd backend
pip install -r requirements.txt

# Required environment variables
export LLM_API_KEY="your-api-key"        # Groq or OpenAI API key (required for embeddings)
export LLM_PROVIDER="groq"               # "groq" or "openai"
export LLM_MODEL="llama3-8b-8192"        # LLM model name
export EMBEDDING_MODEL="text-embedding-ada-002"  # Embedding model name

# Optional environment variables
export SQLITE_DB_PATH="./evilearn.db"
export CHROMA_PERSIST_DIR="./chroma_db"
export TOP_K_RESULTS="5"
export MAX_FILE_SIZE_MB="50"
export CORS_ORIGINS="http://localhost:5173,http://localhost:3000"

# Run the server
python -m backend.main
```

The backend starts on `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend starts on `http://localhost:5173` and proxies `/api` requests to the backend.

### Deployment

| Component | Platform | Notes |
|-----------|----------|-------|
| Frontend | Vercel | Static build via `npm run build` → `dist/` |
| Backend | Render | Python web service, Uvicorn on port 8000 |

## System Guarantees

1. **No claim is verified without document evidence.** The system never uses pre-trained knowledge for verification.
2. **Every verdict is traceable.** Each status links back to specific document snippets and page numbers.
3. **The pipeline order is non-negotiable.** Planner → Claim Extractor → Retriever → Verifier → Explainer.
4. **LLM usage is restricted.** Only the Claim Extractor and Explainer use LLM calls. Verification is purely algorithmic.
5. **Fallback behavior is deterministic.** If the LLM is unavailable, rule-based extraction and explanation are used.
6. **All sessions are audited.** Every input, claim, result, and feedback action is stored in SQLite.
7. **Embeddings come ONLY from the LLM API.** ChromaDB stores vectors, it never generates them.
8. **All data is strictly typed.** Pydantic models validate every claim before storage. No untyped dicts in core flow.
9. **Output is validated before persistence.** Pipeline output is validated via Pydantic schemas BEFORE inserting into the database.

## Limitations

- **PDF only for document upload.** Text files (`.txt`) are also supported, but no other formats (DOCX, HTML, etc.).
- **No real-time document updates.** Documents must be re-uploaded to update the knowledge base.
- **Single-user design.** No authentication or multi-tenancy.
- **Embedding model is fixed at runtime.** Changing the embedding model requires re-processing all documents.
- **No claim deduplication.** Overlapping claims from the same input are not merged.
- **Maximum file size is 50 MB** (configurable via `MAX_FILE_SIZE_MB`).

## Project Structure

```
evilearn/
├── README.md                          # This file
├── backend/
│   ├── README.md                      # Backend documentation
│   ├── app.py                         # FastAPI application
│   ├── config.py                      # Environment configuration
│   ├── schemas.py                     # Pydantic request/response models
│   ├── main.py                        # Uvicorn entry point
│   ├── requirements.txt               # Python dependencies
│   ├── ai_engine/
│   │   ├── README.md                  # AI Engine documentation
│   │   └── pipeline.py                # LangGraph graph-native agents + pipeline
│   └── data_layer/
│       ├── README.md                  # Data Layer documentation
│       ├── document_processor.py      # PDF/text extraction
│       ├── chunker.py                 # Text chunking
│       ├── embedding_service.py       # Embedding generation via LLM API
│       ├── vector_store.py            # ChromaDB interface (storage only)
│       └── database.py                # SQLite interface
└── frontend/
    ├── README.md                      # Frontend documentation
    ├── package.json                   # Node.js dependencies
    ├── vite.config.js                 # Vite configuration
    └── src/
        ├── App.jsx                    # Root component
        ├── api.js                     # API client
        └── components/
            ├── DocumentUpload.jsx     # Document upload interface
            ├── ValidationWorkspace.jsx# Text input & submission
            ├── ResultsDisplay.jsx     # Results summary & claim list
            ├── ClaimCard.jsx          # Individual claim display
            ├── EvidenceViewer.jsx     # Evidence snippet display
            └── HistoryDashboard.jsx   # Session history browser
```
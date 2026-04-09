# EviLearn — Claim-Based Knowledge Validation System

> **EviLearn is NOT a chatbot.** It is a structured, evidence-based reasoning system that validates user-submitted knowledge (answers, summaries, explanations) against uploaded reference documents. Every output is traceable to document evidence.

## Core Mechanism

```
Input → Claims → Evidence → Verification → Stress Test (conditional) → Explanation
```

A user submits text. The system decomposes it into atomic claims, retrieves evidence from uploaded documents, verifies each claim against that evidence, optionally stress-tests reasoning for robustness, and generates a human-readable explanation for every verdict. No claim is accepted or rejected without document-backed reasoning.

## Thinking Simulation Engine

EviLearn includes a **Thinking Simulation Engine** — a LangGraph-based **graph-based cognitive reasoning simulator** that generates structured reasoning graphs for three cognitive levels and performs structural comparison.

### Feature Overview

The Thinking Simulation Engine does **NOT** solve problems or check correctness. It is a **strict cognitive reasoning engine** that:

- **Represents reasoning as graphs** (nodes + edges + decisions), not plain text
- **Enforces cognitive profiles as hard constraints** during generation (allowed/forbidden operations, max abstraction levels)
- **Generates strategies during creation** (not tagged post-hoc)
- **Performs structural comparison** (graph shape, strategy distribution, abstraction flow), not surface-level text comparison
- **Converts student answers to the same graph structure** for precise gap detection

### Architecture (LangGraph Nodes + Flow)

```
START → Cognitive Profile Generator → Parallel Reasoning Generator
→ Reasoning Graph Builder → Strategy-Constrained Generator
→ Abstraction Analyzer → Structural Comparator
→ (if student answer exists) Student Graph Converter → Gap Generator → END
```

All 8 nodes operate as pure functions on a shared `ThinkingState` via LangGraph `StateGraph`. The student graph converter node is conditional — it only executes when a student answer is provided. A validation layer in the Reasoning Graph Builder rejects and regenerates if cognitive constraints are violated.

### Reasoning Graph Schema

Each reasoning path is a true graph, not a step list:

**Nodes:**
- `step_id` — unique identifier
- `operation_type` — what kind of reasoning operation (identify, recall, transform, reduce, etc.)
- `concept_used` — concept or rule applied
- `input` / `output` — data flowing through the step
- `reasoning` — why this step was taken
- `abstraction_level` — LOW, MEDIUM, or HIGH
- `strategy_type` — direct_application, rule_based, transformation, reduction, optimization

**Edges:**
- `from_step_id` → `to_step_id`
- `relation_type` — derives, transforms, or simplifies

**Decisions:**
- `decision_point` — what choice was made
- `alternatives_considered` — what other approaches were available
- `chosen_path_reason` — why this path was selected

### Strict Cognitive Constraints

Profiles act as **hard constraints during generation**, not descriptions:

| Level | Allowed Operations | Forbidden Operations | Max Abstraction |
|-------|-------------------|---------------------|-----------------|
| Beginner | identify, recall, substitute, compute | transform, reframe, abstract, optimize, reduce | LOW |
| Intermediate | analyze, classify, apply_rule, decompose, verify, synthesize | optimize | MEDIUM |
| Expert | transform, reframe, abstract, reduce, optimize (REQUIRED) | none | HIGH |

### Abstraction Model

Each step gets an abstraction score: LOW (1.0), MEDIUM (2.0), HIGH (3.0).

Each path computes:
- **average_abstraction** — mean across all steps
- **max_abstraction** — highest level reached
- **abstraction_transitions** — where level changes (e.g., "e1(HIGH) → e2(HIGH)")
- **abstraction_flow** — sequence of levels (e.g., ["HIGH", "HIGH", "HIGH"])

### Structural Comparison Logic

Comparison is **structural**, not descriptive:

| Dimension | What is Compared |
|-----------|-----------------|
| Graph Shape | node count, edge count, depth, linear vs transformed |
| Strategy Distribution | % of direct_application, rule_based, transformation, reduction, optimization per level |
| Abstraction Flow | average/max abstraction, transitions, flow sequence |
| Key Differences | automatically derived from structural data |

### Validation Rules

After reasoning generation, the system validates:
- Beginner graphs contain no forbidden operations (transform, reframe, etc.)
- Expert graphs contain at least one transformation or reduction step
- Abstraction levels do not exceed profile maximum
- Strategy types are valid (not arbitrary labels)
- Cross-profile: graphs are structurally different (not identical operations)

### State Design

The engine uses a central `ThinkingState` (TypedDict) containing:

| Field | Written By | Description |
|-------|-----------|-------------|
| `problem` | Input | The problem/question to analyze |
| `student_answer` | Input | Optional student reasoning |
| `cognitive_profiles` | Node 1 | Profiles with constraint rules (allowed/forbidden ops, max abstraction) |
| `reasoning_graphs` | Node 2, 3 | Structured graphs with nodes, edges, decisions |
| `strategy_distributions` | Node 4 | Strategy % distributions per level |
| `abstraction_data` | Node 5 | Abstraction metrics per path |
| `comparison_results` | Node 6 | Structural comparison (graph shape, strategy, abstraction) |
| `student_graph` | Node 7 | Student reasoning as graph structure (conditional) |
| `gap_analysis` | Node 8 | Structural gap insights |
| `validation_passed` | Node 3 | Whether constraints were satisfied |
| `validation_notes` | Node 3 | Details of any constraint fixes |

### Module Descriptions

| Node | Name | Responsibility |
|------|------|---------------|
| 1 | Cognitive Profile Generator | Generates 3 profiles with strict constraint rules (allowed/forbidden ops, max abstraction) |
| 2 | Parallel Reasoning Generator | Generates 3 structurally divergent reasoning graphs under profile constraints |
| 3 | Reasoning Graph Builder | Enforces node + edge structure, validates constraints, fixes violations |
| 4 | Strategy-Constrained Generator | Computes strategy distributions from generated strategy_type on each node |
| 5 | Abstraction Analyzer | Computes abstraction metrics: average, max, transitions, flow |
| 6 | Structural Comparator | Compares graphs by shape, strategy distribution, and abstraction flow |
| 7 | Student Graph Converter | Conditional — converts student answer to same graph structure, compares structurally |
| 8 | Gap Generator | Derives gap insights from structural data (step counts, transformation %, abstraction levels) |

### Example Input/Output

**Input:**
```json
{
  "problem": "Calculate the derivative of f(x) = x³ + 2x² - 5x + 3",
  "student_answer": "I used the power rule: f'(x) = 3x² + 4x - 5"
}
```

**Output (abbreviated):**
```json
{
  "cognitive_profiles": [
    {
      "level": "beginner",
      "description": "Applies formulas directly without transformation...",
      "characteristics": ["Direct formula usage", "No transformations"],
      "allowed_operations": ["identify", "recall", "substitute", "compute"],
      "forbidden_operations": ["transform", "reframe", "abstract", "optimize", "reduce"],
      "max_abstraction": "LOW"
    },
    { "level": "intermediate", "max_abstraction": "MEDIUM", ... },
    { "level": "expert", "max_abstraction": "HIGH", ... }
  ],
  "reasoning_graphs": [
    {
      "level": "beginner",
      "nodes": [
        {
          "step_id": "b1",
          "operation_type": "identify",
          "concept_used": "problem recognition",
          "input_value": "f(x) = x³ + 2x² - 5x + 3",
          "output_value": "Identified as polynomial differentiation",
          "reasoning": "Read the problem statement directly",
          "abstraction_level": "LOW",
          "strategy_type": "direct_application"
        },
        { "step_id": "b2", "operation_type": "recall", ... },
        { "step_id": "b3", "operation_type": "substitute", ... },
        { "step_id": "b4", "operation_type": "compute", ... }
      ],
      "edges": [
        { "from_step_id": "b1", "to_step_id": "b2", "relation_type": "derives" },
        { "from_step_id": "b2", "to_step_id": "b3", "relation_type": "derives" },
        { "from_step_id": "b3", "to_step_id": "b4", "relation_type": "derives" }
      ],
      "decisions": [
        {
          "decision_point": "Selected standard formula",
          "alternatives_considered": ["No alternatives considered"],
          "chosen_path_reason": "Used the first formula that came to mind"
        }
      ],
      "abstraction_metrics": {
        "average_abstraction": 1.0,
        "max_abstraction": "LOW",
        "abstraction_transitions": [],
        "abstraction_flow": ["LOW", "LOW", "LOW", "LOW"]
      }
    }
  ],
  "strategy_distributions": [
    {
      "level": "beginner",
      "direct_application_pct": 100.0,
      "rule_based_pct": 0.0,
      "transformation_pct": 0.0,
      "reduction_pct": 0.0,
      "optimization_pct": 0.0,
      "strategies_used": ["direct_application"]
    }
  ],
  "structural_comparison": {
    "graph_shape": {
      "beginner": { "node_count": 4, "edge_count": 3, "depth": 4, "is_linear": true },
      "expert": { "node_count": 4, "edge_count": 3, "depth": 4, "is_linear": false }
    },
    "strategy_distribution": { ... },
    "abstraction_flow": {
      "beginner": { "average_abstraction": 1.0, "max_abstraction": "LOW", "flow": ["LOW", "LOW", "LOW", "LOW"] },
      "expert": { "average_abstraction": 3.0, "max_abstraction": "HIGH", "flow": ["HIGH", "HIGH", "HIGH", "HIGH"] }
    },
    "key_differences": [
      "Beginner uses 4 nodes, intermediate uses 5, expert uses 4 nodes",
      "Beginner follows a linear path; expert uses transformations",
      "Average abstraction: beginner=1.0, expert=3.0 (expert operates at higher abstraction)"
    ]
  },
  "student_graph": {
    "student_level_match": "beginner",
    "nodes": [{ "step_id": "s1", "operation_type": "compute", ... }],
    "edges": [],
    "missing_nodes": ["Missing 'reframe' step (used by expert)"],
    "missing_transformations": ["No 'abstract' transformation (expert uses this)"],
    "abstraction_mismatches": ["Student avg abstraction=1.0 vs expert avg=3.0"]
  },
  "gap_analysis": [
    {
      "insight": "Your approach follows beginner-level reasoning: direct application",
      "severity": "warning",
      "source": "comparison"
    },
    {
      "insight": "Your reasoning contains 0 transformation steps; expert uses 2",
      "severity": "warning",
      "source": "strategy"
    },
    {
      "insight": "Your abstraction level remains LOW throughout; expert shifts to HIGH",
      "severity": "critical",
      "source": "abstraction"
    }
  ],
  "validation_passed": true,
  "validation_notes": []
}
```

### Key Constraints

- Does **NOT** output final answers
- Does **NOT** validate correctness
- Does **NOT** optimize for accuracy of solution
- Focuses **ONLY** on reasoning structure, strategy differences, and abstraction levels

### How to Run and Test

```bash
# Start backend
cd backend
pip install -r requirements.txt
export LLM_API_KEY="your-api-key"
export LLM_PROVIDER="groq"
python -m backend.main

# Start frontend
cd frontend
npm install
npm run dev

# Test the endpoint directly
curl -X POST http://localhost:8000/simulate-thinking \
  -H "Content-Type: application/json" \
  -d '{"problem": "Solve x² - 4 = 0", "student_answer": "x = 2"}'
```

### Tech Stack Justification

| Component | Technology | Reason |
|-----------|-----------|--------|
| Orchestration | LangGraph StateGraph | Mandatory graph-based execution with shared state |
| LLM Layer | OpenAI / Groq | API-based reasoning for profile and path generation |
| Backend | FastAPI | REST API with automatic validation |
| Data Models | Pydantic | Strict typing for all structured outputs |
| State Management | LangGraph State (TypedDict) | Global shared state across all nodes |
| Execution Style | Graph-based only | No sequential function chaining |

## Cognitive Load Optimizer

EviLearn includes a **Cognitive Load Optimizer** — a LangGraph-based **real-time reasoning flow regulator** that controls how explanations are presented to users, adapting structure and pacing to match individual cognitive capacity.

### Feature Overview

The Cognitive Load Optimizer does **NOT** change explanation content or correctness. It is a **presentation control layer** that:

- **Analyzes explanation structure** (steps, concept transitions, abstraction levels)
- **Tracks user cognitive state** dynamically across interactions (understanding, stability, learning speed)
- **Computes cognitive load** from step density, concept gaps, and memory demand
- **Adapts explanation presentation** in real-time (splitting, merging, adjusting abstraction)
- **Runs a cyclic feedback loop** that re-optimizes until load matches capacity

### Why It Exists

Every explanation has a cognitive cost. Too many steps at once → overload. Too few → stagnation. The optimizer continuously enforces **Explanation Complexity ≈ User Capacity**, preventing both overload and underload.

### System Architecture (LangGraph Nodes + Flow)

```
START → Explanation Analyzer → User State Tracker → Load Estimator
→ Control Engine → Granularity Controller → Explanation Restructurer
→ Feedback Loop Manager → (loop back to Load Estimator OR END)
```

All 7 nodes operate as pure functions on a shared `CognitiveLoadState` via LangGraph `StateGraph`. The graph is **cyclic** — the Feedback Loop Manager conditionally routes back to the Load Estimator for re-optimization (up to 3 iterations).

| Node | Name | Responsibility |
|------|------|---------------|
| 1 | Explanation Analyzer | Breaks explanation into steps, extracts concepts, identifies abstraction levels |
| 2 | User State Tracker | Loads/initializes the dynamic user cognitive profile |
| 3 | Load Estimator | Computes cognitive load metrics (step density, concept gap, memory demand) |
| 4 | Control Engine | Compares load vs capacity, decides adaptation strategy (overload/optimal/underload) |
| 5 | Granularity Controller | Adjusts step size — splits, merges, or adds checkpoints based on control decisions |
| 6 | Explanation Restructurer | Validates and cleans restructured steps, ensures dependency consistency |
| 7 | Feedback Loop Manager | Updates user state, determines whether to loop for further optimization |

### Cognitive Load Definition

Cognitive load is a measurable composite quantity derived from three dimensions:

| Dimension | Definition | Measurement |
|-----------|-----------|-------------|
| **Step Density** | How many reasoning steps per unit of content | Steps per 100 words |
| **Concept Gap** | How large the jump between consecutive steps | Average new concepts introduced per transition |
| **Memory Demand** | How many elements must be held simultaneously | Max dependencies + concepts any single step requires |

**Total Load** = `(step_density × 2.0) + (concept_gap × 2.5) + (memory_demand × 1.5)` (capped at 10.0)

### Adaptation Logic

| State | Condition | Behavior |
|-------|-----------|----------|
| **Overload** | load > capacity + 1.5 | Split long steps, reduce abstraction, add intermediate reasoning |
| **Underload** | load < capacity - 2.0 | Merge short steps, compress reasoning, increase abstraction |
| **Optimal** | within range | Maintain structure, add checkpoints if borderline |

Reasoning modes map to adaptation:
- **fine-grained** → step-by-step mode (overload state)
- **medium** → grouped reasoning (optimal state)
- **coarse** → compressed reasoning (underload state)

### Feedback Loop

The user cognitive state is **dynamic and evolves** after every interaction:

| Field | Updated When | Direction |
|-------|-------------|-----------|
| `understanding_level` | Every interaction | Decreases on overload, increases on underload |
| `reasoning_stability` | Every interaction | Decreases on overload, increases otherwise |
| `learning_speed` | Optimal reached | Increases when optimal state is achieved |
| `overload_signals` | Overload detected | Increments on overload, decrements on underload |
| `interaction_count` | Always | Increments by 1 |

The feedback loop re-runs load estimation after each adaptation. It converges when either:
- Load state becomes "optimal"
- Maximum iterations (3) are reached

### Pipeline Integration

```
User Query → Reasoning System → Explanation Generator → Cognitive Load Optimizer → Final Output
```

The optimizer sits at the end of the pipeline, between raw explanation output and the user. It receives the full explanation and reshapes it without changing content.

### Cognitive Load Output Contract

The `/optimize-cognitive-load` endpoint returns:

```json
{
  "adapted_explanation": [
    {
      "step_id": "s1",
      "content": "First, identify the problem type...",
      "concepts": ["problem recognition"],
      "abstraction_level": "concrete",
      "depends_on": []
    }
  ],
  "load_state": "optimal",
  "control_actions": [
    {
      "action": "maintain",
      "reason": "Load matches capacity — maintaining current structure"
    }
  ],
  "user_state": {
    "user_id": "default",
    "understanding_level": 0.5,
    "reasoning_stability": 0.52,
    "learning_speed": 0.52,
    "overload_signals": 0,
    "interaction_count": 1
  },
  "load_metrics": {
    "step_density": 3.33,
    "concept_gap": 1.0,
    "memory_demand": 2.0,
    "total_load": 5.16
  },
  "reasoning_mode": "medium"
}
```


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
        ST[Stress Test Engine<br/>conditional]
        EA[Explanation Agent]
        CLO[Cognitive Load Optimizer<br/>cyclic feedback]
        PL --> CE --> RA --> VA --> ST --> EA --> CLO
    end

    subgraph "Layer 4 — Data & Knowledge Layer"
        VEC[(ChromaDB<br/>Vector Store<br/>storage only)]
        SQL[(SQLite<br/>Relational DB)]
        DP[Document Processor<br/>PyMuPDF]
        EMB[EmbeddingService<br/>LLM API]
    end

    UI <-->|REST API| API
    API --> PL
    CLO --> API
    EA --> API
    ST --> API
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

    Note over PIP: Stage 5 (conditional) — Stress Test
    PIP->>PIP: If run_stress_test: analyze reasoning robustness

    Note over PIP: Stage 6 — Explainer
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

### Stress Test Flow

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI
    participant PIP as Stress Test Engine

    U->>FE: Submit problem + answer + confidence
    FE->>API: POST /evaluate-reasoning {problem, student_answer, confidence}
    API->>PIP: pipeline.evaluate_reasoning(problem, student_answer, confidence)

    Note over PIP: Extract concepts from claims
    Note over PIP: Extract hidden assumptions (LLM + rules)
    Note over PIP: Extract constraints from problem/answer
    Note over PIP: Detect reasoning weaknesses
    Note over PIP: Generate edge cases (hybrid)
    Note over PIP: Generate adversarial scenarios
    Note over PIP: Evaluate reasoning under each scenario
    Note over PIP: Compute robustness score
    Note over PIP: Convert failures to adversarial questions
    Note over PIP: Format structured output

    PIP-->>API: Stress test results
    API-->>FE: EvaluateReasoningResponse
    FE-->>U: Robustness score, failures, weaknesses, questions
```

### Thinking Simulation Flow

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI
    participant TSE as Thinking Simulation Engine

    U->>FE: Submit problem + optional student answer
    FE->>API: POST /simulate-thinking {problem, student_answer?}
    API->>TSE: thinking_engine.simulate(problem, student_answer)

    Note over TSE: Node 1: Generate cognitive profiles with constraint rules
    Note over TSE: Node 2: Generate 3 structurally divergent reasoning graphs
    Note over TSE: Node 3: Validate + enforce constraints (reject/fix violations)
    Note over TSE: Node 4: Compute strategy distributions from generated strategy_types
    Note over TSE: Node 5: Compute abstraction metrics (avg, max, transitions, flow)
    Note over TSE: Node 6: Structural comparison (graph shape, strategy %, abstraction)
    Note over TSE: Node 7 (conditional): Convert student answer to graph structure
    Note over TSE: Node 8: Generate structural gap insights

    TSE-->>API: Thinking simulation results
    API-->>FE: ThinkingSimulationResponse
    FE-->>U: Reasoning graphs, strategy distributions, structural comparison, gap analysis
```

### Cognitive Load Optimization Flow

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI
    participant CLO as Cognitive Load Optimizer

    U->>FE: Submit explanation + user ID
    FE->>API: POST /optimize-cognitive-load {explanation, user_id}
    API->>CLO: cognitive_load_optimizer.optimize(explanation, user_id)

    Note over CLO: Node 1: Analyze explanation into steps, concepts, abstraction
    Note over CLO: Node 2: Load/initialize user cognitive state
    Note over CLO: Node 3: Compute cognitive load (step density, concept gap, memory)
    Note over CLO: Node 4: Compare load vs capacity → overload/optimal/underload
    Note over CLO: Node 5: Adjust granularity (split/merge/checkpoint)
    Note over CLO: Node 6: Restructure and validate adapted steps
    Note over CLO: Node 7: Update user state, decide loop/end

    alt Load not optimal & iterations remaining
        Note over CLO: Loop back to Node 3 (re-estimate load)
    end

    CLO-->>API: Adapted explanation + load state + user state
    API-->>FE: CognitiveLoadResponse
    FE-->>U: Adapted steps, load state badges, control actions, user metrics
```

## User Flow

1. **Upload Documents** — User uploads PDF or TXT files that form the knowledge base.
2. **Enter Text** — User submits an answer, summary, or explanation to validate.
3. **View Results** — System displays each extracted claim with a status badge, confidence score, evidence snippets, and explanation.
4. **Provide Feedback** — User can accept, reject, or edit individual claims.
5. **Review History** — User can browse past validation sessions with full results.
6. **Stress Test Reasoning** — User submits a problem and student answer with confidence level. System stress-tests the reasoning for edge cases, weaknesses, and adversarial scenarios, returning a robustness score and targeted questions.
7. **Thinking Simulation** — User submits a problem (and optionally a student answer). System simulates beginner, intermediate, and expert reasoning paths, performs structural comparison, and identifies thinking gaps.
8. **Cognitive Load Optimization** — User submits an explanation to optimize. System analyzes cognitive load (step density, concept gaps, memory demand), compares against user capacity, and adapts presentation — splitting, merging, or adjusting abstraction in real-time with a cyclic feedback loop.

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
| `POST` | `/evaluate-reasoning` | Stress-test reasoning robustness | `{problem, student_answer, confidence}` | `EvaluateReasoningResponse` |
| `POST` | `/simulate-thinking` | Simulate multi-level cognitive reasoning | `{problem, student_answer?}` | `ThinkingSimulationResponse` |
| `POST` | `/optimize-cognitive-load` | Optimize explanation cognitive load | `{explanation, user_id?}` | `CognitiveLoadResponse` |

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

### Stress Test Output Contract

The `/evaluate-reasoning` endpoint returns this structure:

```json
{
  "stress_test_results": [
    "FAILS when: x = 0 (at: division step) — Division by zero"
  ],
  "weakness_summary": [
    {
      "type": "overgeneralization",
      "detail": "Assumes all values are positive without justification"
    }
  ],
  "robustness_summary": {
    "robustness_score": 0.4,
    "summary": "Reasoning fails under multiple edge cases",
    "level": "low"
  },
  "adversarial_questions": [
    "What happens when x = 0?"
  ]
}
```

### Thinking Simulation Output Contract

The `/simulate-thinking` endpoint returns this structure:

```json
{
  "cognitive_profiles": [
    {
      "level": "beginner",
      "description": "Applies formulas directly...",
      "characteristics": ["Direct formula usage", "No transformations"],
      "allowed_operations": ["identify", "recall", "substitute", "compute"],
      "forbidden_operations": ["transform", "reframe", "abstract", "optimize", "reduce"],
      "max_abstraction": "LOW"
    }
  ],
  "reasoning_graphs": [
    {
      "level": "beginner",
      "nodes": [
        {
          "step_id": "b1",
          "operation_type": "identify",
          "concept_used": "problem recognition",
          "input_value": "Problem text",
          "output_value": "Identified problem type",
          "reasoning": "Read the problem statement directly",
          "abstraction_level": "LOW",
          "strategy_type": "direct_application"
        }
      ],
      "edges": [
        { "from_step_id": "b1", "to_step_id": "b2", "relation_type": "derives" }
      ],
      "decisions": [
        {
          "decision_point": "Selected standard formula",
          "alternatives_considered": ["No alternatives considered"],
          "chosen_path_reason": "Used the first formula that came to mind"
        }
      ],
      "abstraction_metrics": {
        "average_abstraction": 1.0,
        "max_abstraction": "LOW",
        "abstraction_transitions": [],
        "abstraction_flow": ["LOW", "LOW"]
      },
      "metadata": {
        "step_count": 4,
        "edge_count": 3,
        "has_transformation": false
      }
    }
  ],
  "strategy_distributions": [
    {
      "level": "beginner",
      "direct_application_pct": 100.0,
      "rule_based_pct": 0.0,
      "transformation_pct": 0.0,
      "reduction_pct": 0.0,
      "optimization_pct": 0.0,
      "strategies_used": ["direct_application"]
    }
  ],
  "structural_comparison": {
    "graph_shape": {
      "beginner": { "node_count": 4, "edge_count": 3, "depth": 4, "is_linear": true }
    },
    "strategy_distribution": {
      "beginner": { "direct_application_pct": 100.0, "transformation_pct": 0.0 }
    },
    "abstraction_flow": {
      "beginner": { "average_abstraction": 1.0, "max_abstraction": "LOW", "flow": ["LOW"] }
    },
    "key_differences": [
      "Beginner uses 4 nodes; expert uses 4 nodes",
      "Beginner follows a linear path; expert uses transformations"
    ]
  },
  "gap_analysis": [
    {
      "insight": "Your approach follows beginner-level reasoning: direct application",
      "severity": "warning",
      "source": "comparison"
    }
  ],
  "student_graph": {
    "student_level_match": "beginner",
    "nodes": [],
    "edges": [],
    "abstraction_metrics": { "average_abstraction": 1.0, "max_abstraction": "LOW" },
    "missing_nodes": [],
    "missing_transformations": [],
    "unnecessary_steps": [],
    "abstraction_mismatches": [],
    "strategy_distribution": {}
  },
  "validation_passed": true,
  "validation_notes": []
}
```

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
3. **The pipeline order is non-negotiable.** Planner → Claim Extractor → Retriever → Verifier → Stress Test (conditional) → Explainer.
4. **LLM usage is restricted.** Only the Claim Extractor and Explainer use LLM calls. Verification is purely algorithmic.
5. **Fallback behavior is deterministic.** If the LLM is unavailable, rule-based extraction and explanation are used.
6. **All sessions are audited.** Every input, claim, result, and feedback action is stored in SQLite.
7. **Embeddings come ONLY from the LLM API.** ChromaDB stores vectors, it never generates them.
8. **All data is strictly typed.** Pydantic models validate every claim before storage. No untyped dicts in core flow.
9. **Output is validated before persistence.** Pipeline output is validated via Pydantic schemas BEFORE inserting into the database.
10. **Thinking Simulation is graph-based.** The Thinking Simulation Engine uses LangGraph StateGraph with 7 nodes, conditional edges, and shared state — no sequential function chaining.
11. **Cognitive Load Optimizer is cyclic.** The Cognitive Load Optimizer uses LangGraph StateGraph with 7 nodes and a cyclic feedback loop that re-optimizes until load matches user capacity. It never changes explanation content — only structure and pacing.

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
│   │   ├── pipeline.py                # LangGraph graph-native agents + pipeline
│   │   ├── thinking_engine.py         # Thinking Simulation Engine (LangGraph)
│   │   ├── cognitive_load_optimizer.py# Cognitive Load Optimizer (LangGraph, cyclic)
│   │   └── stress_test_agent/         # Knowledge Stress-Test Engine
│   │       ├── stress_test_agent.py   # Main orchestrator
│   │       ├── concept_extractor.py   # Extract key concepts from claims
│   │       ├── assumption_extractor.py# Extract hidden assumptions (LLM + rules)
│   │       ├── constraint_extractor.py# Extract constraints from problem/answer
│   │       ├── weakness_analyzer.py   # Detect reasoning weaknesses
│   │       ├── edge_case_generator.py # Generate boundary/edge cases (hybrid)
│   │       ├── adversarial_engine.py  # Generate adversarial scenarios
│   │       ├── failure_analyzer.py    # Evaluate reasoning under scenarios
│   │       ├── robustness_evaluator.py# Compute robustness score
│   │       ├── adversarial_question_agent.py # Convert failures to questions
│   │       └── output_formatter.py    # Format final structured output
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
            ├── HistoryDashboard.jsx   # Session history browser
            ├── StressTestWorkspace.jsx# Stress test input form
            ├── StressTestResults.jsx  # Stress test results display
            ├── ThinkingSimulationWorkspace.jsx # Thinking simulation input
│             ├── ThinkingSimulationResults.jsx   # Thinking simulation results
│             ├── CognitiveLoadWorkspace.jsx      # Cognitive load optimizer input
│             └── CognitiveLoadResults.jsx        # Cognitive load optimizer results
```
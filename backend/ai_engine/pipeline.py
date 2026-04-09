"""EviLearn Validation Pipeline — Graph-native multi-agent system using LangGraph.

ALL agent logic is implemented as pure functions (graph nodes) operating on
shared LangGraph state. There are NO agent classes. Each node reads from state,
performs its single responsibility, and writes ONLY its assigned fields.

Pipeline order (NON-NEGOTIABLE):
    START → planner → claim_extractor → retriever → verifier → explainer → END

Strict typing:
    All data flowing through state uses Pydantic models from schemas.py.
    No untyped dicts in the core data flow.

Embedding:
    Embeddings are generated via EmbeddingService (LLM API).
    ChromaDB is used ONLY for storage and similarity search.

Tech stack: LangGraph StateGraph + LLM API (Groq/OpenAI) — nothing else.
"""

import os
import json
import re
import uuid
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END

from ..schemas import (
    ClaimItem,
    EvidenceItem,
    EvidenceChunk,
    VerificationResult,
    FinalClaimResult,
)
from .stress_test_agent import run_stress_test


# ---------------------------------------------------------------------------
# Shared State — every node reads from / writes to this TypedDict
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    """Shared state flowing through every node in the LangGraph pipeline.

    Dependencies are injected once before graph.invoke() and never mutated.
    Pipeline data fields are each written by exactly one node.
    All data uses strict Pydantic models — no untyped dicts.
    """

    # Injected dependencies (set once before graph.invoke, never mutated by nodes)
    _vector_store: object
    _llm_client: object
    _embedding_service: object

    # Pipeline data (each field written by exactly one node)
    raw_input: str
    input_type: str                          # written by planner
    pipeline_decision: str                   # written by planner
    claims: list[dict]                       # written by claim_extractor (ClaimItem dicts)
    evidence_map: dict                       # written by retriever (claim_id → list[EvidenceChunk dicts])
    verification_results: list[dict]         # written by verifier (VerificationResult dicts)
    final_results: list[dict]                # written by explainer (FinalClaimResult dicts)
    error: Optional[str]

    # Stress test fields
    problem: str                             # optional problem context
    run_stress_test: bool                    # flag to enable stress testing
    stress_test_output: dict                 # written by stress_test_node


# ---------------------------------------------------------------------------
# Node 1: Planner — detect input type, decide pipeline routing
# ---------------------------------------------------------------------------

def planner_node(state: PipelineState) -> dict:
    """Determine input type (answer / explanation / summary / question).

    Reads: raw_input
    Writes: input_type, pipeline_decision
    """
    raw_input = state["raw_input"]
    if not raw_input or not raw_input.strip():
        raise ValueError("Input text is empty.")

    text_lower = raw_input.strip().lower()

    # Detect question
    if text_lower.endswith("?"):
        input_type = "question"
    else:
        question_starters = [
            "what ", "how ", "why ", "when ", "where ",
            "who ", "which ", "is ", "are ", "do ",
            "does ", "can ", "could ",
        ]
        input_type = "question" if any(
            text_lower.startswith(s) for s in question_starters
        ) else None

    # Detect explanation
    if input_type is None:
        explanation_keywords = [
            "because", "therefore", "this means", "the reason",
            "this is due to", "as a result", "consequently",
        ]
        if any(kw in text_lower for kw in explanation_keywords):
            input_type = "explanation"

    # Detect summary
    if input_type is None:
        summary_keywords = [
            "in summary", "to summarize", "overall",
            "in conclusion", "the main points",
        ]
        if any(kw in text_lower for kw in summary_keywords):
            input_type = "summary"

    # Default to answer
    if input_type is None:
        input_type = "answer"

    return {
        "input_type": input_type,
        "pipeline_decision": "validation",
    }


# ---------------------------------------------------------------------------
# Node 2: Claim Extractor — break text into atomic factual claims
# ---------------------------------------------------------------------------

def claim_extractor_node(state: PipelineState) -> dict:
    """Extract atomic factual claims from user input.

    Uses LLM API when available, falls back to rule-based sentence splitting.
    Output is validated via ClaimItem Pydantic model.

    Reads: raw_input, input_type, _llm_client
    Writes: claims (list of ClaimItem dicts)
    """
    text = state["raw_input"].strip()
    input_type = state["input_type"]
    llm_client = state.get("_llm_client")

    if not text:
        return {"claims": []}

    # --- Try LLM-based extraction first ---
    if llm_client:
        try:
            prompt = (
                f"Break the following {input_type} into atomic factual claims.\n"
                "Each claim must:\n"
                "- Represent a single fact\n"
                "- Be independently verifiable\n"
                "- Preserve the original meaning\n\n"
                "Return ONLY a JSON array of strings, each being one claim.\n\n"
                f"Text: {text}\n\nClaims:"
            )
            response = llm_client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                claims_list = json.loads(json_match.group())
                claims = []
                for c in claims_list:
                    if isinstance(c, str) and c.strip():
                        item = ClaimItem(
                            claim_id=str(uuid.uuid4()),
                            claim_text=c.strip(),
                        )
                        claims.append(item.model_dump())
                return {"claims": claims}
        except Exception:
            pass  # Fall through to rule-based extraction

    # --- Rule-based fallback: sentence splitting ---
    sentences = re.split(r'(?<=[.!])\s+', text)
    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        if sentence.endswith("?"):
            continue
        item = ClaimItem(
            claim_id=str(uuid.uuid4()),
            claim_text=sentence,
        )
        claims.append(item.model_dump())

    return {"claims": claims}


# ---------------------------------------------------------------------------
# Conditional edge: check if claims were extracted
# ---------------------------------------------------------------------------

def check_claims_extracted(state: PipelineState) -> str:
    """Route to retriever if claims exist, otherwise end."""
    if state.get("claims"):
        return "has_claims"
    return "no_claims"


# ---------------------------------------------------------------------------
# Node 3: Retriever — retrieve document evidence for each claim
# ---------------------------------------------------------------------------

def retriever_node(state: PipelineState) -> dict:
    """Retrieve top-k evidence chunks from ChromaDB for each claim.

    Embeddings are generated via EmbeddingService (LLM API).
    ChromaDB is used ONLY for similarity search with pre-computed embeddings.
    Uses ONLY retrieved documents. Does NOT generate evidence.

    Reads: claims, _vector_store, _embedding_service
    Writes: evidence_map (claim_id → list of EvidenceChunk dicts)
    """
    vector_store = state.get("_vector_store")
    embedding_service = state.get("_embedding_service")
    claims = state["claims"]
    evidence_map: dict = {}

    for claim in claims:
        claim_id = claim["claim_id"]
        claim_text = claim["claim_text"]
        try:
            # Generate query embedding via LLM API
            query_embedding = embedding_service.embed_query(claim_text)
            # Query ChromaDB with pre-computed embedding
            raw_evidence = vector_store.query(
                query_embedding=query_embedding, top_k=5
            )
            # Validate each evidence chunk via Pydantic
            validated = []
            for e in raw_evidence:
                chunk = EvidenceChunk(
                    text_snippet=e.get("text_snippet", ""),
                    page_number=e.get("page_number", 0),
                    relevance_score=e.get("relevance_score", 0.0),
                    document_id=e.get("document_id", ""),
                )
                validated.append(chunk.model_dump())
            evidence_map[claim_id] = validated
        except Exception:
            evidence_map[claim_id] = []

    return {"evidence_map": evidence_map}


# ---------------------------------------------------------------------------
# Node 4: Verifier — evaluate claim status using only evidence relevance
# ---------------------------------------------------------------------------

# Thresholds aligned with system prompt confidence scale
_HIGH_THRESHOLD = 0.7
_MEDIUM_THRESHOLD = 0.4


def verifier_node(state: PipelineState) -> dict:
    """Assign status and confidence to each claim based solely on evidence.

    Output is validated via VerificationResult Pydantic model.

    Status definitions:
        supported       — evidence clearly confirms the claim
        weakly_supported — partial or indirect support
        unsupported     — no supporting evidence or contradiction

    Confidence scale:
        High  (0.8–1.0)  strong direct match
        Medium (0.4–0.7)  partial match
        Low   (0.0–0.3)  no support or weak match

    Reads: claims, evidence_map
    Writes: verification_results (list of VerificationResult dicts)
    """
    claims = state["claims"]
    evidence_map = state["evidence_map"]
    results: list[dict] = []

    for claim in claims:
        claim_id = claim["claim_id"]
        claim_text = claim["claim_text"]
        evidence_list = evidence_map.get(claim_id, [])

        # --- Determine status from highest relevance score ---
        if not evidence_list:
            status = "unsupported"
            confidence_score = 0.1
        else:
            max_relevance = max(
                e.get("relevance_score", 0.0) for e in evidence_list
            )
            if max_relevance >= _HIGH_THRESHOLD:
                status = "supported"
                confidence_score = round(min(max_relevance, 1.0), 2)
            elif max_relevance >= _MEDIUM_THRESHOLD:
                status = "weakly_supported"
                confidence_score = round(max_relevance, 2)
            else:
                status = "unsupported"
                confidence_score = round(max(max_relevance, 0.05), 2)

        # Build evidence snippets (top 3), validated via EvidenceItem
        evidence_snippets = [
            EvidenceItem(
                snippet=e.get("text_snippet", ""),
                page_number=e.get("page_number", 0),
            ).model_dump()
            for e in evidence_list[:3]
        ]

        # Validate via VerificationResult
        vr = VerificationResult(
            claim_id=claim_id,
            claim_text=claim_text,
            status=status,
            confidence_score=confidence_score,
            evidence=[EvidenceItem(**e) for e in evidence_snippets],
        )
        results.append(vr.model_dump())

    return {"verification_results": results}


# ---------------------------------------------------------------------------
# Node 5: Stress Test — break student reasoning under adversarial conditions
# ---------------------------------------------------------------------------

def stress_test_node(state: PipelineState) -> dict:
    """Run the Knowledge Stress-Test Engine on verified claims.

    This is NOT a tutoring feature. It actively tries to break the
    student's reasoning by generating adversarial scenarios, detecting
    failures, and producing targeted challenge questions.

    Reads: raw_input, problem, claims, verification_results, _llm_client
    Writes: stress_test_output
    """
    student_answer = state["raw_input"]
    claims = state.get("claims", [])
    verification_results = state.get("verification_results", [])
    problem = state.get("problem", "")
    llm_client = state.get("_llm_client")

    result = run_stress_test(
        student_answer=student_answer,
        claims=claims,
        verification_results=verification_results,
        problem=problem,
        llm_client=llm_client,
    )

    return {"stress_test_output": result}


def _check_stress_test(state: PipelineState) -> str:
    """Route to stress_test if enabled, else skip to explainer."""
    if state.get("run_stress_test"):
        return "run_stress_test"
    return "skip"


# ---------------------------------------------------------------------------
# Node 6: Explainer — generate human-readable explanations
# ---------------------------------------------------------------------------

def explainer_node(state: PipelineState) -> dict:
    """Generate an explanation for each verification result.

    Uses LLM API when available, falls back to rule-based templates.
    Does NOT change the verification decision.
    Output is validated via FinalClaimResult Pydantic model.

    Reads: verification_results, _llm_client
    Writes: final_results (list of FinalClaimResult dicts)
    """
    llm_client = state.get("_llm_client")
    verification_results = state["verification_results"]
    final_results: list[dict] = []

    for result in verification_results:
        explanation = _explain_with_llm(result, llm_client) if llm_client else None
        if explanation is None:
            explanation = _explain_with_rules(result)

        # Validate via FinalClaimResult
        fcr = FinalClaimResult(
            claim_id=result["claim_id"],
            claim_text=result["claim_text"],
            status=result["status"],
            confidence_score=result["confidence_score"],
            evidence=[EvidenceItem(**e) for e in result.get("evidence", [])],
            explanation=explanation,
        )
        final_results.append(fcr.model_dump())

    return {"final_results": final_results}


def _explain_with_llm(result: dict, llm_client) -> Optional[str]:
    """Generate explanation using LLM API. Returns None on failure."""
    try:
        evidence_text = ""
        for i, e in enumerate(result.get("evidence", []), 1):
            evidence_text += (
                f"\n  Evidence {i} (page {e.get('page_number', '?')}): "
                f"{e.get('snippet', '')}"
            )

        prompt = (
            f"Explain why this claim has the status '{result['status']}' "
            f"with confidence {result['confidence_score']}.\n\n"
            f"Claim: {result['claim_text']}\n"
            f"Status: {result['status']}\n"
            f"Confidence: {result['confidence_score']}\n"
            f"Evidence: {evidence_text}\n\n"
            "Write a concise explanation (2-3 sentences) that:\n"
            "1. References the evidence\n"
            "2. Explains why the claim has this status\n"
            "3. Does NOT change the decision\n\n"
            "Explanation:"
        )
        response = llm_client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def _explain_with_rules(result: dict) -> str:
    """Generate a rule-based explanation from the verification result."""
    status = result.get("status", "unsupported")
    confidence = result.get("confidence_score", 0)
    evidence = result.get("evidence", [])

    if status == "supported":
        if evidence:
            page_refs = ", ".join(
                f"page {e.get('page_number', '?')}" for e in evidence
            )
            return (
                f"This claim is supported by evidence found in the uploaded "
                f"documents ({page_refs}). The retrieved content closely "
                f"matches the assertion with a confidence score of {confidence}."
            )
        return f"This claim is marked as supported with confidence {confidence}."

    if status == "weakly_supported":
        if evidence:
            page_refs = ", ".join(
                f"page {e.get('page_number', '?')}" for e in evidence
            )
            return (
                f"This claim has partial support from the documents "
                f"({page_refs}). The evidence provides indirect or incomplete "
                f"confirmation with a confidence score of {confidence}."
            )
        return f"This claim has weak support with confidence {confidence}."

    # unsupported
    if evidence:
        return (
            "Despite retrieving potentially related content, the evidence "
            f"does not sufficiently support this claim. The confidence "
            f"score is {confidence}."
        )
    return (
        "No supporting evidence was found in the uploaded documents for "
        f"this claim. The confidence score is {confidence}."
    )


# ---------------------------------------------------------------------------
# Graph Construction — builds the LangGraph StateGraph
# ---------------------------------------------------------------------------

def build_validation_graph():
    """Build and compile the LangGraph StateGraph.

    Returns a compiled graph with strict sequential execution:
        START → planner → claim_extractor → [retriever → verifier
        → (stress_test)? → explainer] → END

    The stress_test node is conditional — it runs only when
    run_stress_test is True in state.

    All nodes are pure functions. No classes. No wrappers.
    """
    graph = StateGraph(PipelineState)

    # Register pure-function nodes
    graph.add_node("planner", planner_node)
    graph.add_node("claim_extractor", claim_extractor_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("stress_test", stress_test_node)
    graph.add_node("explainer", explainer_node)

    # Strict sequential edges (NON-NEGOTIABLE order)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "claim_extractor")
    graph.add_conditional_edges(
        "claim_extractor",
        check_claims_extracted,
        {"has_claims": "retriever", "no_claims": END},
    )
    graph.add_edge("retriever", "verifier")
    # After verifier, conditionally route to stress_test or explainer
    graph.add_conditional_edges(
        "verifier",
        _check_stress_test,
        {"run_stress_test": "stress_test", "skip": "explainer"},
    )
    graph.add_edge("stress_test", "explainer")
    graph.add_edge("explainer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API — used by the backend to execute the pipeline
# ---------------------------------------------------------------------------

class ValidationPipeline:
    """Thin entry point that holds runtime dependencies and invokes the graph.

    This is NOT an agent class. It only:
    1. Stores references to vector_store, llm_client, embedding_service
    2. Injects them into shared state
    3. Calls graph.invoke()

    All reasoning logic lives in the graph nodes above.
    """

    def __init__(self, vector_store, llm_client=None, embedding_service=None):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embedding_service = embedding_service
        self.graph = build_validation_graph()

    def execute(self, raw_input: str) -> dict:
        """Execute the LangGraph validation pipeline.

        Args:
            raw_input: User's input text.

        Returns:
            Dict with input_type and structured claim results.
            All claim data is validated via FinalClaimResult before return.

        Raises:
            ValueError: If input is invalid.
            RuntimeError: If a pipeline stage fails.
        """
        if not raw_input or not raw_input.strip():
            raise ValueError("Input text is empty.")

        initial_state: PipelineState = {
            "_vector_store": self.vector_store,
            "_llm_client": self.llm_client,
            "_embedding_service": self.embedding_service,
            "raw_input": raw_input,
            "input_type": "",
            "pipeline_decision": "",
            "claims": [],
            "evidence_map": {},
            "verification_results": [],
            "final_results": [],
            "error": None,
            "problem": "",
            "run_stress_test": False,
            "stress_test_output": {},
        }

        final_state = self.graph.invoke(initial_state)

        claims = final_state.get("final_results", [])
        if not claims:
            return {
                "input_type": final_state.get("input_type", "answer"),
                "claims": [],
                "message": "No factual claims could be extracted from the input.",
            }

        # Final validation — every claim must pass FinalClaimResult validation
        validated_claims = []
        for c in claims:
            validated = FinalClaimResult(**c)
            validated_claims.append(validated.model_dump())

        return {
            "input_type": final_state.get("input_type", "answer"),
            "claims": validated_claims,
        }

    def evaluate_reasoning(
        self, student_answer: str, problem: str = ""
    ) -> dict:
        """Execute the pipeline with stress testing enabled.

        Runs the full pipeline (planner → claim_extractor → retriever
        → verifier → stress_test → explainer) and returns the stress
        test output.

        Args:
            student_answer: The student's answer to stress-test.
            problem: Optional problem statement for context.

        Returns:
            Dict with stress_test_results, weakness_summary,
            robustness_summary, adversarial_questions.

        Raises:
            ValueError: If input is invalid.
            RuntimeError: If a pipeline stage fails.
        """
        if not student_answer or not student_answer.strip():
            raise ValueError("Student answer is empty.")

        initial_state: PipelineState = {
            "_vector_store": self.vector_store,
            "_llm_client": self.llm_client,
            "_embedding_service": self.embedding_service,
            "raw_input": student_answer,
            "input_type": "",
            "pipeline_decision": "",
            "claims": [],
            "evidence_map": {},
            "verification_results": [],
            "final_results": [],
            "error": None,
            "problem": problem,
            "run_stress_test": True,
            "stress_test_output": {},
        }

        final_state = self.graph.invoke(initial_state)

        return final_state.get("stress_test_output", {})


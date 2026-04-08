"""Validation Pipeline — Orchestrates the multi-agent reasoning pipeline using LangGraph."""

from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, START, END

from .agents.planner import PlannerAgent
from .agents.claim_extractor import ClaimExtractionAgent
from .agents.retriever import RetrievalAgent
from .agents.verifier import VerificationAgent
from .agents.explainer import ExplanationAgent


class PipelineState(TypedDict):
    """State that flows through the LangGraph pipeline stages."""

    raw_input: str
    input_type: str
    pipeline_decision: str
    claims: list[dict]
    evidence_map: dict
    verification_results: list[dict]
    final_results: list[dict]
    error: Optional[str]


class ValidationPipeline:
    """Executes the strict 5-stage validation pipeline using LangGraph StateGraph.

    Pipeline order (NON-NEGOTIABLE):
    1. Planner Agent
    2. Claim Extraction Agent
    3. Retrieval Agent
    4. Verification Agent
    5. Explanation Agent

    Each stage is a node in the LangGraph StateGraph, connected in strict sequence.
    """

    def __init__(self, vector_store, llm_client=None):
        """Initialize pipeline with required services.

        Args:
            vector_store: VectorStore instance.
            llm_client: Optional LLM client for agents that use it.
        """
        self.planner = PlannerAgent()
        self.claim_extractor = ClaimExtractionAgent(llm_client=llm_client)
        self.retriever = RetrievalAgent(vector_store=vector_store)
        self.verifier = VerificationAgent()
        self.explainer = ExplanationAgent(llm_client=llm_client)

        # Build the LangGraph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph StateGraph with strict pipeline order.

        Returns:
            Compiled LangGraph ready for execution.
        """
        graph = StateGraph(PipelineState)

        # Add nodes (each agent is a node)
        graph.add_node("planner", self._run_planner)
        graph.add_node("claim_extractor", self._run_claim_extractor)
        graph.add_node("retriever", self._run_retriever)
        graph.add_node("verifier", self._run_verifier)
        graph.add_node("explainer", self._run_explainer)

        # Define strict sequential edges (NON-NEGOTIABLE order)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "claim_extractor")
        graph.add_conditional_edges(
            "claim_extractor",
            self._check_claims_extracted,
            {"has_claims": "retriever", "no_claims": END},
        )
        graph.add_edge("retriever", "verifier")
        graph.add_edge("verifier", "explainer")
        graph.add_edge("explainer", END)

        return graph.compile()

    # --- Node functions ---

    def _run_planner(self, state: PipelineState) -> dict:
        """Stage 1: Planner Agent — Determine input type."""
        try:
            plan = self.planner.run(state["raw_input"])
            return {
                "input_type": plan["input_type"],
                "pipeline_decision": plan["pipeline_decision"],
            }
        except Exception as e:
            raise ValueError(f"Planner stage failed: {e}")

    def _run_claim_extractor(self, state: PipelineState) -> dict:
        """Stage 2: Claim Extraction Agent — Break text into atomic claims."""
        try:
            claims = self.claim_extractor.run(
                text=state["raw_input"].strip(),
                input_type=state["input_type"],
            )
            return {"claims": claims}
        except Exception as e:
            raise RuntimeError(f"Claim extraction stage failed: {e}")

    def _check_claims_extracted(self, state: PipelineState) -> str:
        """Conditional edge: check if claims were extracted."""
        if state.get("claims"):
            return "has_claims"
        return "no_claims"

    def _run_retriever(self, state: PipelineState) -> dict:
        """Stage 3: Retrieval Agent — Retrieve evidence for each claim."""
        try:
            evidence_map = self.retriever.run(
                claims=state["claims"],
                top_k=5,
            )
            return {"evidence_map": evidence_map}
        except Exception as e:
            raise RuntimeError(f"Retrieval stage failed: {e}")

    def _run_verifier(self, state: PipelineState) -> dict:
        """Stage 4: Verification Agent — Evaluate claims against evidence."""
        try:
            verification_results = self.verifier.run(
                claims=state["claims"],
                evidence_map=state["evidence_map"],
            )
            return {"verification_results": verification_results}
        except Exception as e:
            raise RuntimeError(f"Verification stage failed: {e}")

    def _run_explainer(self, state: PipelineState) -> dict:
        """Stage 5: Explanation Agent — Generate explanations."""
        try:
            final_results = self.explainer.run(state["verification_results"])
            return {"final_results": final_results}
        except Exception as e:
            raise RuntimeError(f"Explanation stage failed: {e}")

    # --- Public API ---

    def execute(self, raw_input: str) -> dict:
        """Execute the full validation pipeline via LangGraph.

        Args:
            raw_input: User's input text.

        Returns:
            Dict with input_type and structured claim results.

        Raises:
            ValueError: If input is invalid.
            RuntimeError: If a pipeline stage fails.
        """
        if not raw_input or not raw_input.strip():
            raise ValueError("Input text is empty.")

        # Initialize state
        initial_state: PipelineState = {
            "raw_input": raw_input,
            "input_type": "",
            "pipeline_decision": "",
            "claims": [],
            "evidence_map": {},
            "verification_results": [],
            "final_results": [],
            "error": None,
        }

        # Execute the LangGraph pipeline
        final_state = self.graph.invoke(initial_state)

        # Build response
        claims = final_state.get("final_results", [])
        if not claims:
            return {
                "input_type": final_state.get("input_type", "answer"),
                "claims": [],
                "message": "No factual claims could be extracted from the input.",
            }

        return {
            "input_type": final_state.get("input_type", "answer"),
            "claims": claims,
        }

"""Validation Pipeline — Orchestrates the multi-agent reasoning pipeline using LangGraph concepts."""

import os
from typing import Optional

from .agents.planner import PlannerAgent
from .agents.claim_extractor import ClaimExtractionAgent
from .agents.retriever import RetrievalAgent
from .agents.verifier import VerificationAgent
from .agents.explainer import ExplanationAgent


class PipelineState:
    """Holds the state that flows through the pipeline stages."""

    def __init__(self):
        self.raw_input: str = ""
        self.input_type: str = ""
        self.pipeline_decision: str = ""
        self.claims: list[dict] = []
        self.evidence_map: dict = {}
        self.verification_results: list[dict] = []
        self.final_results: list[dict] = []
        self.error: Optional[str] = None


class ValidationPipeline:
    """Executes the strict 5-stage validation pipeline.

    Pipeline order (NON-NEGOTIABLE):
    1. Planner Agent
    2. Claim Extraction Agent
    3. Retrieval Agent
    4. Verification Agent
    5. Explanation Agent
    """

    def __init__(self, vector_store, embedding_service, llm_client=None):
        """Initialize pipeline with required services.

        Args:
            vector_store: VectorStore instance.
            embedding_service: EmbeddingService instance.
            llm_client: Optional LLM client for agents that use it.
        """
        self.planner = PlannerAgent()
        self.claim_extractor = ClaimExtractionAgent(llm_client=llm_client)
        self.retriever = RetrievalAgent(vector_store=vector_store, embedding_service=embedding_service)
        self.verifier = VerificationAgent()
        self.explainer = ExplanationAgent(llm_client=llm_client)

    def execute(self, raw_input: str) -> dict:
        """Execute the full validation pipeline.

        Args:
            raw_input: User's input text.

        Returns:
            Dict with session info and structured results.

        Raises:
            ValueError: If input is invalid.
            RuntimeError: If a pipeline stage fails.
        """
        state = PipelineState()
        state.raw_input = raw_input

        # Stage 1: Planner
        try:
            plan = self.planner.run(raw_input)
            state.input_type = plan["input_type"]
            state.pipeline_decision = plan["pipeline_decision"]
        except Exception as e:
            raise ValueError(f"Planner stage failed: {e}")

        # Stage 2: Claim Extraction
        try:
            state.claims = self.claim_extractor.run(
                text=plan["original_input"],
                input_type=state.input_type,
            )
        except Exception as e:
            raise RuntimeError(f"Claim extraction stage failed: {e}")

        if not state.claims:
            return {
                "input_type": state.input_type,
                "claims": [],
                "message": "No factual claims could be extracted from the input.",
            }

        # Stage 3: Retrieval
        try:
            state.evidence_map = self.retriever.run(
                claims=state.claims,
                top_k=5,
            )
        except Exception as e:
            raise RuntimeError(f"Retrieval stage failed: {e}")

        # Stage 4: Verification
        try:
            state.verification_results = self.verifier.run(
                claims=state.claims,
                evidence_map=state.evidence_map,
            )
        except Exception as e:
            raise RuntimeError(f"Verification stage failed: {e}")

        # Stage 5: Explanation
        try:
            state.final_results = self.explainer.run(state.verification_results)
        except Exception as e:
            raise RuntimeError(f"Explanation stage failed: {e}")

        return {
            "input_type": state.input_type,
            "claims": state.final_results,
        }

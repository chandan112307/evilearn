"""Planner Agent — Determines input type and routes the validation pipeline."""

import os
import json
import re
from typing import Optional


class PlannerAgent:
    """Determines input type and decides pipeline routing.
    
    Detects whether input is an answer, explanation, summary, or question.
    Routes to validation pipeline unless input is invalid.
    """

    VALID_INPUT_TYPES = ["answer", "explanation", "summary", "question"]

    def run(self, raw_input: str) -> dict:
        """Analyze input and determine pipeline decision.

        Args:
            raw_input: The user's submitted text.

        Returns:
            Dict with input_type and pipeline_decision.

        Raises:
            ValueError: If input is empty or invalid.
        """
        if not raw_input or not raw_input.strip():
            raise ValueError("Input text is empty.")

        input_type = self._detect_input_type(raw_input)
        pipeline_decision = "validation"

        return {
            "input_type": input_type,
            "pipeline_decision": pipeline_decision,
            "original_input": raw_input.strip(),
        }

    def _detect_input_type(self, text: str) -> str:
        """Heuristically detect the type of input.

        Args:
            text: Input text to classify.

        Returns:
            One of: answer, explanation, summary, question.
        """
        text_lower = text.strip().lower()

        # Check if it's a question
        if text_lower.endswith("?"):
            return "question"

        question_starters = ["what ", "how ", "why ", "when ", "where ", "who ", "which ", "is ", "are ", "do ", "does ", "can ", "could "]
        for starter in question_starters:
            if text_lower.startswith(starter):
                return "question"

        # Check for explanation indicators
        explanation_keywords = ["because", "therefore", "this means", "the reason", "this is due to", "as a result", "consequently"]
        for kw in explanation_keywords:
            if kw in text_lower:
                return "explanation"

        # Check for summary indicators
        summary_keywords = ["in summary", "to summarize", "overall", "in conclusion", "the main points"]
        for kw in summary_keywords:
            if kw in text_lower:
                return "summary"

        return "answer"

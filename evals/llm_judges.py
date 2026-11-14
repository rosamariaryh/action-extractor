"""Gemini-backed LLM judges for transcript extraction evaluations."""

from __future__ import annotations

import atexit
import json
import os
from dataclasses import replace
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field
from opentelemetry import trace

from app.llm_connection import GeminiLLM
from app.llm_connection.client import create_llm
from app.llm_connection.settings import load_llm_settings
from observability.phoenix import setup_phoenix_tracing
from evals.recording import record_evaluation_result
from evals.results import EvaluationResult


class ActionabilityVerdict(BaseModel):
    """Structured score and explanation returned by Gemini for a judge call."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="1.0 means clear owner, concrete task, and real commitment; 0.0 means vague or false positive.",
    )
    label: Literal["high_actionability", "vague", "false_positive", "true_negative"]
    explanation: str = Field(description="Brief justification for the score")


@lru_cache(maxsize=1)
def _get_judge_client() -> GeminiLLM:
    """Create one configured Gemini client for repeated evaluation calls."""
    settings = load_llm_settings()
    judge_model = os.getenv("GEMINI_JUDGE_MODEL_NAME", "").strip()
    if judge_model:
        settings = replace(settings, model=judge_model)

    client = create_llm(settings)
    atexit.register(client.close)
    return client


def actionability_judge(
    input: dict[str, Any],
    output: dict[str, Any],
    expected: dict[str, Any],
) -> EvaluationResult:
    """Judge whether extracted action items are grounded, actionable commitments."""
    setup_phoenix_tracing()
    tracer = trace.get_tracer("action-extractor.evals")
    with tracer.start_as_current_span("eval.actionability") as span:
        action_items = output.get("action_items", [])
        expected_count = expected.get("expected_action_count")

        if not action_items and expected_count == 0:
            verdict = ActionabilityVerdict(
                score=1.0,
                label="true_negative",
                explanation="Correctly extracted zero tasks from a transcript with no expected actions.",
            )
        else:
            prompt = f"""Evaluate the extracted action items against the raw meeting transcript.

Score from 0.0 to 1.0:
- 1.0: The extracted items are genuine commitments with concrete tasks and clear owners where stated. Obvious commitments are not missed.
- 0.5: The tasks are real commitments but are vague, incomplete, or missing an owner that the transcript provides.
- 0.0: The output treats jokes, sarcasm, hypotheticals, or vague social chatter as work; invents commitments; or misses the clear expected actions.

Treat the expected count as evaluation guidance, not as evidence for an item. Judge each item against the transcript. Do not reward guessed owners or details.

Expected action count: {expected_count if expected_count is not None else "not provided"}
Expected notes: {expected.get("notes", "N/A")}

Transcript segments:
{json.dumps(input.get("segments", []), ensure_ascii=False, indent=2)}

Extracted action items:
{json.dumps(action_items, ensure_ascii=False, indent=2)}
"""

            verdict = _get_judge_client().generate_structured(
                prompt,
                response_schema=ActionabilityVerdict,
                system_instruction=(
                    "You are a strict, impartial evaluator for a meeting action-item extractor. "
                    "Return a score based only on the supplied transcript and extraction."
                ),
            )

        span.set_attributes(
            {
                "eval.name": "actionability",
                "eval.type": "llm_judge",
                "eval.score": verdict.score,
                "eval.label": verdict.label,
                "eval.explanation": verdict.explanation,
            }
        )
        result = EvaluationResult(
            score=verdict.score,
            label=verdict.label,
            explanation=verdict.explanation,
        )
        span_id = f"{span.get_span_context().span_id:016x}"

    record_evaluation_result("actionability", "llm_judge", result, span_id)
    return result

"""Run deterministic and Gemini evaluations for one transcript JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from opentelemetry import trace

from app.processing import process_file
from evals.deterministic_eval import quote_grounding, valid_json_schema, valid_timestamps
from evals.llm_judges import actionability_judge
from evals.recording import begin_evaluation_run, record_evaluation_result
from evals.results import EvaluationResult
from observability.phoenix import setup_phoenix_tracing


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "input1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval_output"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _run_deterministic_evaluation(
    name: str,
    evaluator: Callable[..., EvaluationResult],
    transcript: dict[str, Any],
    extracted: dict[str, Any],
    expected: dict[str, Any],
) -> EvaluationResult:
    tracer = trace.get_tracer("action-extractor.evals")
    with tracer.start_as_current_span(f"eval.{name}") as span:
        if name == "valid_json_schema":
            result = evaluator(extracted)
        else:
            result = evaluator(transcript, extracted)
        span.set_attributes(
            {
                "eval.name": name,
                "eval.type": "deterministic_metric",
                "eval.score": result.score,
                "eval.label": result.label,
                "eval.explanation": result.explanation,
            }
        )
        span_id = f"{span.get_span_context().span_id:016x}"
    record_evaluation_result(name, "deterministic_metric", result, span_id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run extraction, deterministic checks, and a Gemini actionability judge."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Transcript JSON path (default: output/input1.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save a new extraction (default: eval_output/)",
    )
    parser.add_argument(
        "--expected-action-count",
        type=int,
        help="Optional expected action-item count to guide the judge",
    )
    parser.add_argument(
        "--notes",
        help="Optional evaluation notes to guide the judge",
    )
    args = parser.parse_args()

    transcript = _read_json(args.input)
    # If the selected file already contains extraction fields, judge it directly.
    # Otherwise, run the app's normal extraction and save its result first.
    if {"decisions", "action_items", "follow_ups"}.issubset(transcript):
        extraction_path = args.input
        extracted = transcript
    else:
        extraction_path = process_file(args.input, args.output_dir)
        extracted = _read_json(extraction_path)

    expected = transcript.get("expected", {})
    if not isinstance(expected, dict):
        expected = {}
    if args.expected_action_count is not None:
        expected["expected_action_count"] = args.expected_action_count
    if args.notes is not None:
        expected["notes"] = args.notes

    judge_input = {
        key: value
        for key, value in transcript.items()
        if key not in {"decisions", "action_items", "follow_ups", "expected"}
    }
    setup_phoenix_tracing()
    evaluation_log_path = begin_evaluation_run(args.input, judge_input, expected)
    results = [
        (
            name,
            _run_deterministic_evaluation(
                name, evaluator, judge_input, extracted, expected
            ),
        )
        for name, evaluator in (
            ("valid_json_schema", valid_json_schema),
            ("valid_timestamps", valid_timestamps),
            ("quote_grounding", quote_grounding),
        )
    ]
    results.append(("actionability", actionability_judge(judge_input, extracted, expected)))

    print(f"Extraction: {extraction_path}")
    print("\nDeterministic metrics:")
    for name, result in results[:-1]:
        score = f"{result.score:.2f}" if result.score is not None else "unscored"
        print(f"  {name}: {score} ({result.label}) — {result.explanation}")
    print("\nLLM judges:")
    for name, result in results[-1:]:
        score = f"{result.score:.2f}" if result.score is not None else "unscored"
        print(f"  {name}: {score} ({result.label}) — {result.explanation}")
    print(f"Evaluation log: {evaluation_log_path}")


if __name__ == "__main__":
    main()

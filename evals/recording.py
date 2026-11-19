"""Persist evaluator results locally and attach them to Phoenix spans."""

from __future__ import annotations

import json
import logging
import os
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from httpx import HTTPStatusError
from phoenix.client import Client

from opentelemetry import trace

from evals.results import EvaluationResult


_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_RUN_STATE: ContextVar[tuple[Path, dict[str, Any]] | None] = ContextVar(
    "eval_run_state", default=None
)
_OUTPUT_LOCK = Lock()
_LOGGER = logging.getLogger(__name__)


def begin_evaluation_run(
    input_file: Path | None,
    input_data: dict[str, Any],
    expected: dict[str, Any],
) -> Path:
    """Create a timestamped JSON report with document details stored once."""
    output_dir = Path(os.getenv("EVAL_OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_file = output_dir / f"evaluations_{timestamp}.json"
    suffix = 1
    while True:
        try:
            output_file.touch(exist_ok=False)
            break
        except FileExistsError:
            output_file = output_dir / f"evaluations_{timestamp}_{suffix}.json"
            suffix += 1
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input": {
            "file": str(input_file.resolve()) if input_file is not None else None,
            "transcript": input_data,
        },
        "expected": expected,
        "metrics": [],
    }
    _RUN_STATE.set((output_file, report))
    _write_report(output_file, report)
    return output_file


def record_evaluation_result(
    name: str,
    evaluator_type: str,
    result: EvaluationResult,
    span_id: str,
) -> None:
    """Add one metric result to the run report and upload its Phoenix annotation."""
    state = _RUN_STATE.get()
    if state is None:
        begin_evaluation_run(None, {}, {})
        state = _RUN_STATE.get()
        assert state is not None
    output_file, report = state
    report["metrics"].append(
        {
            "type": evaluator_type,
            "name": name,
            "span_id": span_id,
            "result": {
                "score": result.score,
                "label": result.label,
                "explanation": result.explanation,
            },
        }
    )
    _write_report(output_file, report)
    _log_to_phoenix(f"{evaluator_type}.{name}", span_id, result)


def _write_report(output_file: Path, report: dict[str, Any]) -> None:
    """Write a complete, indented JSON report after each metric finishes."""
    temporary_file = output_file.with_suffix(".tmp")
    with _OUTPUT_LOCK:
        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_file.replace(output_file)


def _log_to_phoenix(name: str, span_id: str, result: EvaluationResult) -> None:
    collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    endpoint = os.getenv("PHOENIX_UI_ENDPOINT", "").strip()
    if not endpoint and collector_endpoint:
        endpoint = collector_endpoint.replace(":4317", ":6006").rstrip("/")
    if not endpoint:
        return

    if span_id == "0000000000000000":
        _LOGGER.warning("Evaluation saved locally but not attached to Phoenix: tracing is disabled")
        return

    try:
        provider = trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if force_flush and not force_flush():
            _LOGGER.warning(
                "Evaluation saved locally but not attached to Phoenix: trace export did not flush"
            )
            return

        api_key = os.getenv("PHOENIX_API_KEY", "").strip()
        if api_key and not os.getenv("PHOENIX_CLIENT_HEADERS"):
            os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={api_key}"

        client = Client(base_url=endpoint)
        delays = (0.1, 0.25, 0.5, 1.0, 2.0)
        for attempt, delay in enumerate((*delays, None)):
            try:
                client.spans.add_span_annotation(
                    span_id=span_id,
                    annotation_name=name,
                    annotator_kind="LLM" if name.startswith("llm_judge.") else "CODE",
                    score=result.score,
                    label=result.label,
                    explanation=result.explanation,
                    sync=True,
                )
                break
            except HTTPStatusError as exc:
                # Phoenix accepts OTLP spans asynchronously. A brief delay can occur
                # between flushing a span and making it available for annotations.
                if exc.response.status_code != 404 or delay is None:
                    raise
                time.sleep(delay)
    except Exception:
        _LOGGER.exception("Could not attach %s evaluation result to Phoenix", name)

import json
import re

from pydantic import ValidationError

from app.schemas import GroundedItem
from evals.results import EvaluationResult

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$")


def valid_json_schema(output: str | dict) -> EvaluationResult:
    """Validate the current extraction output and every extracted item."""
    try:
        data = json.loads(output) if isinstance(output, str) else output
        if not isinstance(data, dict):
            raise ValueError("Output must be a JSON object")
        required_keys = {"decisions", "action_items", "follow_ups"}
        missing = required_keys - set(data.keys())
        if missing:
            return EvaluationResult(
                score=0.0,
                label="invalid_schema",
                explanation=f"Missing required keys: {sorted(missing)}",
            )
        for category in sorted(required_keys):
            if not isinstance(data[category], list):
                return EvaluationResult(
                    score=0.0,
                    label="invalid_schema",
                    explanation=f"'{category}' must be a list",
                )
            for item in data[category]:
                GroundedItem.model_validate(item)
        return EvaluationResult(
            score=1.0,
            label="valid",
            explanation="Output contains decisions, action_items, and follow_ups with valid grounded items",
        )
    except (ValueError, TypeError, ValidationError) as exc:
        return EvaluationResult(score=0.0, label="invalid_schema", explanation=str(exc))


def valid_timestamps(input: dict, output: dict) -> EvaluationResult:
    """Check timestamp format and require each range to match a source segment."""
    segments = input.get("segments", [])
    if not isinstance(segments, list) or not segments:
        return EvaluationResult(score=0.0, label="no_segments", explanation="Empty input segments")

    source_ranges = [
        (segment.get("start"), segment.get("end"))
        for segment in segments
        if isinstance(segment, dict)
    ]
    items = _all_items(output)
    if not items:
        return EvaluationResult(score=1.0, label="pass_empty", explanation="No items to validate")

    for item in items:
        start, end = item.get("start", ""), item.get("end", "")
        if (
            not isinstance(start, str)
            or not isinstance(end, str)
            or not _valid_timestamp(start)
            or not _valid_timestamp(end)
        ):
            return EvaluationResult(
                score=0.0,
                label="bad_format",
                explanation=f"Invalid start/end timestamp: {start!r}, {end!r}",
            )
        start_seconds, end_seconds = _timestamp_seconds(start), _timestamp_seconds(end)
        in_source_segment = any(
            isinstance(segment_start, str)
            and isinstance(segment_end, str)
            and _valid_timestamp(segment_start)
            and _valid_timestamp(segment_end)
            and _timestamp_seconds(segment_start) <= start_seconds <= end_seconds
            and end_seconds <= _timestamp_seconds(segment_end)
            for segment_start, segment_end in source_ranges
        )
        if not in_source_segment:
            return EvaluationResult(
                score=0.0,
                label="out_of_bounds",
                explanation=f"Timestamp range [{start}, {end}] falls outside every source segment",
            )

    return EvaluationResult(
        score=1.0,
        label="valid",
        explanation=f"All {len(items)} timestamp ranges fall within source segments",
    )


def quote_grounding(input: dict, output: dict) -> EvaluationResult:
    """3. Deterministic anti-hallucination check: verifies every extracted quote exists in the transcript."""
    segments = input.get("segments", [])
    source_texts = [
        segment.get("text", "").casefold()
        for segment in segments
        if isinstance(segment, dict) and isinstance(segment.get("text", ""), str)
    ]
    items = _all_items(output)

    if not items:
        return EvaluationResult(score=1.0, label="grounded", explanation="No claims to verify")

    ungrounded = []
    for item in items:
        quote = item.get("quote", "")
        if not isinstance(quote, str) or not quote.strip():
            ungrounded.append("missing quote")
        elif not any(quote.casefold() in text for text in source_texts):
            ungrounded.append(repr(quote))

    if ungrounded:
        score = max(0.0, 1.0 - (len(ungrounded) / len(items)))
        return EvaluationResult(
            score=round(score, 2),
            label="hallucinated",
            explanation=f"Quotes not found verbatim in any source segment: {'; '.join(ungrounded)}",
        )

    return EvaluationResult(
        score=1.0,
        label="grounded",
        explanation=f"All {len(items)} quotes were found verbatim in source segments",
    )


def _all_items(output: dict) -> list[dict]:
    """Return all current extraction categories, ignoring malformed collections."""
    items = []
    for category in ("decisions", "action_items", "follow_ups"):
        category_items = output.get(category, [])
        if isinstance(category_items, list):
            items.extend(item for item in category_items if isinstance(item, dict))
    return items


def _timestamp_seconds(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _valid_timestamp(timestamp: str) -> bool:
    if not TIMESTAMP_RE.fullmatch(timestamp):
        return False
    _, minutes, seconds = timestamp.split(":")
    return int(minutes) < 60 and float(seconds) < 60

"""Speaker-turn-aware chunking for diarized transcript JSON."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


Transcript = Mapping[str, Any]
TranscriptChunk = dict[str, Any]


def chunk_transcript(
    transcript: Transcript,
    *,
    max_chars: int = 12_000,
) -> list[TranscriptChunk]:
    """Split a transcript into LLM-sized JSON objects at speaker-turn boundaries.

    Each output retains the transcript metadata and complete segment objects,
    including speaker labels and timestamp strings. ``max_chars`` is measured
    against compact JSON serialization; an individual turn larger than the
    limit is kept intact in its own chunk rather than losing its timing data.

    Args:
        transcript: Mapping with the transcript metadata and a ``segments`` list.
        max_chars: Target maximum serialized size for each chunk. Must be positive.

    Returns:
        JSON-compatible transcript chunks in original segment order. An empty
        segment list produces no chunks.

    Raises:
        ValueError: If the limit is invalid or a transcript segment lacks the
            speaker, start, end, or text fields required for grounded extraction.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    segments = transcript.get("segments")
    if not isinstance(segments, list):
        raise ValueError("transcript must contain a 'segments' list")
    if not segments:
        return []

    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError(f"segments[{index}] must be an object")
        missing = {"speaker", "start", "end", "text"} - segment.keys()
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"segments[{index}] is missing required fields: {fields}")
        if not isinstance(segment["text"], str) or not segment["text"].strip():
            raise ValueError(f"segments[{index}].text must be a non-empty string")

    metadata = {key: value for key, value in transcript.items() if key != "segments"}
    segment_groups: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []

    def serialized_size(group: list[Mapping[str, Any]]) -> int:
        # Reserve space for the chunk bookkeeping fields added in the result.
        candidate = {
            **metadata,
            "chunk_index": 999_999_999,
            "chunk_count": 999_999_999,
            "segments": group,
        }
        return len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")))

    for segment in segments:
        candidate = [*current, segment]
        if current and serialized_size(candidate) > max_chars:
            segment_groups.append(current)
            current = [segment]
        else:
            current = candidate

    if current:
        segment_groups.append(current)

    count = len(segment_groups)
    return [
        {
            **metadata,
            "chunk_index": index,
            "chunk_count": count,
            "segments": group,
        }
        for index, group in enumerate(segment_groups, start=1)
    ]

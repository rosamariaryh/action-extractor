"""Transcript extraction and output file handling."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from opentelemetry import trace

from app.chunker import chunk_transcript
from app.llm_connection import GeminiLLM
from app.llm_connection.client import create_llm
from app.prompts import EXTRACTION_SYSTEM_INSTRUCTION, build_chunk_prompt
from app.llm_connection.rate_limiter import RequestRateLimiter
from app.schemas import ChunkExtraction
from app.llm_connection.settings import LLMSettings, load_llm_settings


def extract_transcript(
    transcript: dict[str, Any],
    *,
    llm: GeminiLLM,
    settings: LLMSettings,
    max_chars: int = 12_000,
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract each transcript chunk and append results to the input."""
    chunks = chunks if chunks is not None else chunk_transcript(transcript, max_chars=max_chars)
    results = ChunkExtraction()
    rate_limiter = RequestRateLimiter(settings.requests_per_minute)
    tracer = trace.get_tracer(__name__)

    def extract_chunk(chunk: dict[str, Any]) -> ChunkExtraction:
        with tracer.start_as_current_span("transcript.extract.chunk") as span:
            span.set_attribute("transcript.id", str(transcript.get("transcript_id", "")))
            span.set_attribute("transcript.chunk.index", chunk.get("chunk_index", 1))
            span.set_attribute("transcript.chunk.count", chunk.get("chunk_count", 1))
            span.set_attribute("llm.provider", settings.provider)
            span.set_attribute("llm.model", settings.model)

            for attempt in range(settings.max_retries + 1):
                rate_limiter.acquire()
                try:
                    return llm.generate_structured(
                        build_chunk_prompt(chunk),
                        response_schema=ChunkExtraction,
                        system_instruction=EXTRACTION_SYSTEM_INSTRUCTION,
                    )
                except Exception as exc:
                    if not llm.is_retryable_error(exc) or attempt == settings.max_retries:
                        raise
                    time.sleep(min(2**attempt, 30))
            raise RuntimeError("Unreachable retry state")

    with ThreadPoolExecutor(max_workers=settings.max_concurrency) as executor:
        for extraction in executor.map(extract_chunk, chunks):
            results.decisions.extend(extraction.decisions)
            results.action_items.extend(extraction.action_items)
            results.follow_ups.extend(extraction.follow_ups)

    return {
        **transcript,
        "decisions": [item.model_dump(mode="json") for item in results.decisions],
        "action_items": [item.model_dump(mode="json") for item in results.action_items],
        "follow_ups": [item.model_dump(mode="json") for item in results.follow_ups],
    }


def process_transcript(
    transcript: dict[str, Any],
    *,
    model: str | None = None,
    max_chars: int = 12_000,
    settings: LLMSettings | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run extraction for an in-memory transcript and close the LLM client."""
    settings = settings or load_llm_settings()
    if model is not None:
        settings = replace(settings, model=model)

    llm = create_llm(settings)
    try:
        return extract_transcript(
            transcript,
            llm=llm,
            settings=settings,
            max_chars=max_chars,
            chunks=chunks,
        )
    finally:
        llm.close()


def process_file(
    input_path: Path,
    output_dir: Path,
    *,
    model: str | None = None,
    max_chars: int = 12_000,
) -> Path:
    """Read a transcript JSON file and save its extraction in the output folder."""
    with input_path.open("r", encoding="utf-8") as input_file:
        transcript = json.load(input_file)
    if not isinstance(transcript, dict):
        raise ValueError("Input JSON must contain a transcript object")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = process_transcript(
        transcript,
        model=model,
        max_chars=max_chars,
    )

    output_path = output_dir / input_path.name
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return output_path

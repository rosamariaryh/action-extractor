"""FastAPI HTTP interface for transcript extraction."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.chunker import chunk_transcript
from app.processing import process_transcript
from app.schemas import GroundedItem
from app.llm_connection.settings import load_llm_settings


app = FastAPI(
    title="Action Extractor API",
    description="Extract grounded decisions, action items, and follow-ups from diarized transcripts.",
    version="0.1.0",
)


class ExtractionResponse(BaseModel):
    """Extraction arrays plus the original transcript fields at the top level."""

    model_config = ConfigDict(extra="allow")

    decisions: list[GroundedItem] = Field(default_factory=list)
    action_items: list[GroundedItem] = Field(default_factory=list)
    follow_ups: list[GroundedItem] = Field(default_factory=list)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Return a simple process health status."""
    return {"status": "ok"}


@app.post(
    "/extract",
    tags=["extraction"],
    response_model=ExtractionResponse,
    responses={
        502: {"description": "Transcript extraction failed"},
        503: {"description": "LLM provider is not configured"},
    },
)
def extract(transcript: dict[str, Any]) -> ExtractionResponse:
    """Extract grounded items from a transcript JSON request body."""
    try:
        chunks = chunk_transcript(transcript)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        settings = load_llm_settings()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="LLM provider is not configured") from exc

    try:
        result = process_transcript(transcript, settings=settings, chunks=chunks)
        return ExtractionResponse.model_validate(result)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Transcript extraction failed") from exc

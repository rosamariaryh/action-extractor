"""Pydantic schemas for structured LLM extraction results."""

from pydantic import BaseModel, Field


class GroundedItem(BaseModel):
    """An extracted item tied to one exact source segment."""

    description: str = Field(description="Concise description of the extracted item")
    owner: str | None = Field(
        default=None,
        description="Person responsible, only when explicitly identified",
    )
    due_date: str | None = Field(
        default=None,
        description="Deadline as stated, or null when none is stated",
    )
    quote: str = Field(description="Exact, contiguous quote copied from the source segment")
    speaker: str = Field(description="Speaker label from the source segment")
    start: str = Field(description="Exact start timestamp from the source segment")
    end: str = Field(description="Exact end timestamp from the source segment")


class ChunkExtraction(BaseModel):
    """Structured output expected from Gemini for a single transcript chunk."""

    decisions: list[GroundedItem] = Field(default_factory=list)
    action_items: list[GroundedItem] = Field(default_factory=list)
    follow_ups: list[GroundedItem] = Field(default_factory=list)

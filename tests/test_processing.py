from unittest.mock import patch

import pytest
from google.genai.errors import ClientError

from app.llm_connection import GeminiLLM
from app.llm_connection.settings import LLMSettings
from app.processing import extract_transcript
from app.schemas import ChunkExtraction, GroundedItem


class FakeLLM:
    def __init__(self, errors: list[Exception] | None = None) -> None:
        self.errors = list(errors or [])
        self.calls = 0

    @staticmethod
    def is_retryable_error(error: Exception) -> bool:
        return GeminiLLM.is_retryable_error(error)

    def generate_structured(
        self,
        prompt: str,
        *,
        response_schema: type[ChunkExtraction],
        system_instruction: str | None = None,
    ) -> ChunkExtraction:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return ChunkExtraction(
            decisions=[
                GroundedItem(
                    description=f"Decision {self.calls}",
                    quote=f"Turn {self.calls}.",
                    speaker="Speaker 0",
                    start=f"00:00:0{self.calls}.000",
                    end=f"00:00:0{self.calls + 1}.000",
                )
            ]
        )


def transcript_with_two_turns() -> dict[str, object]:
    return {
        "transcript_id": "processing-test",
        "segments": [
            {"speaker": "Speaker 0", "start": "00:00:01.000", "end": "00:00:02.000", "text": "Turn one."},
            {"speaker": "Speaker 1", "start": "00:00:03.000", "end": "00:00:04.000", "text": "Turn two."},
        ],
    }


@pytest.fixture
def settings() -> LLMSettings:
    return LLMSettings(
        provider="gemini",
        api_key="unused-in-unit-test",
        model="fake-model",
        max_concurrency=1,
        max_retries=1,
        requests_per_minute=120,
    )


def test_aggregates_results_from_all_chunks_and_preserves_input(settings: LLMSettings) -> None:
    llm = FakeLLM()

    with patch("app.processing.time.sleep"):
        result = extract_transcript(
            transcript_with_two_turns(),
            llm=llm,  # type: ignore[arg-type]
            settings=settings,
            max_chars=1,
        )

    assert result["transcript_id"] == "processing-test"
    assert [item["description"] for item in result["decisions"]] == ["Decision 1", "Decision 2"]
    assert result["action_items"] == []
    assert result["follow_ups"] == []
    assert llm.calls == 2


def test_retries_rate_limit_but_does_not_retry_bad_request(settings: LLMSettings) -> None:
    retrying_llm = FakeLLM([ClientError(429, {})])
    with patch("app.processing.time.sleep"):
        extract_transcript(
            transcript_with_two_turns(),
            llm=retrying_llm,  # type: ignore[arg-type]
            settings=settings,
            max_chars=1,
        )
    assert retrying_llm.calls == 3

    bad_request_llm = FakeLLM([ClientError(400, {})])
    one_turn = transcript_with_two_turns()
    one_turn["segments"] = one_turn["segments"][:1]
    with pytest.raises(ClientError):
        extract_transcript(
            one_turn,
            llm=bad_request_llm,  # type: ignore[arg-type]
            settings=settings,
            max_chars=1,
        )
    assert bad_request_llm.calls == 1

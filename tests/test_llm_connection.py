from types import SimpleNamespace

import pytest

from app.llm_connection import GeminiLLM
from app.schemas import ChunkExtraction, GroundedItem


class FakeModels:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.call = None

    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.call = kwargs
        return self.response


class FakeClient:
    def __init__(self, response: SimpleNamespace) -> None:
        self.models = FakeModels(response)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def sample_extraction() -> ChunkExtraction:
    return ChunkExtraction(
        decisions=[
            GroundedItem(
                description="Move launch to Tuesday",
                quote="Let's launch Tuesday.",
                speaker="Speaker 0",
                start="00:00:01.000",
                end="00:00:03.000",
            )
        ]
    )


def test_sends_structured_generation_request_and_returns_parsed_model() -> None:
    expected = sample_extraction()
    client = FakeClient(SimpleNamespace(parsed=expected, text=None))
    llm = GeminiLLM(model="test-model", client=client)  # type: ignore[arg-type]

    actual = llm.generate_structured(
        "Extract decisions.",
        response_schema=ChunkExtraction,
        system_instruction="Only use transcript evidence.",
    )

    assert actual == expected
    assert client.models.call["model"] == "test-model"
    assert client.models.call["contents"] == "Extract decisions."
    config = client.models.call["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is ChunkExtraction
    assert config.system_instruction == "Only use transcript evidence."

    llm.close()
    assert client.closed


def test_validates_json_when_sdk_does_not_return_parsed_model() -> None:
    expected = sample_extraction()
    client = FakeClient(SimpleNamespace(parsed=None, text=expected.model_dump_json()))
    llm = GeminiLLM(client=client)  # type: ignore[arg-type]

    actual = llm.generate_structured("Extract.", response_schema=ChunkExtraction)

    assert actual == expected


def test_rejects_blank_prompt_before_request() -> None:
    client = FakeClient(SimpleNamespace(parsed=sample_extraction(), text=None))
    llm = GeminiLLM(client=client)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="prompt must not be empty"):
        llm.generate_structured("  ", response_schema=ChunkExtraction)
    assert client.models.call is None

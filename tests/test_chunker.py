import pytest

from app.chunker import chunk_transcript


@pytest.fixture
def transcript() -> dict[str, object]:
    return {
        "transcript_id": "test-transcript",
        "language": "en",
        "segments": [
            {
                "speaker": f"Speaker {index % 2}",
                "start": f"00:00:{index * 10:02d}.000",
                "end": f"00:00:{index * 10 + 5:02d}.000",
                "text": f"Turn {index}: " + ("meeting details " * 8),
            }
            for index in range(3)
        ],
    }


def test_splits_at_turn_boundaries_and_preserves_order_and_metadata(transcript: dict[str, object]) -> None:
    chunks = chunk_transcript(transcript, max_chars=300)

    assert len(chunks) > 1
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk["chunk_count"] == len(chunks) for chunk in chunks)
    assert all(chunk["transcript_id"] == "test-transcript" for chunk in chunks)
    assert [segment for chunk in chunks for segment in chunk["segments"]] == transcript["segments"]


def test_keeps_a_single_oversized_turn_intact(transcript: dict[str, object]) -> None:
    chunks = chunk_transcript(transcript, max_chars=1)

    assert len(chunks) == len(transcript["segments"])
    assert [chunk["segments"][0] for chunk in chunks] == transcript["segments"]


def test_rejects_segments_without_grounding_fields() -> None:
    with pytest.raises(ValueError, match="missing required fields: end"):
        chunk_transcript({"segments": [{"speaker": "Speaker 0", "start": "00:00:00", "text": "Hi"}]})

"""Gemini API wrapper used by the action extraction application."""

from __future__ import annotations

from typing import TypeVar
from pydantic import BaseModel

try:
    from google import genai
    from google.genai import errors
    from google.genai import types
except ImportError as exc:  # pragma: no cover - depends on optional environment setup
    raise ImportError(
        "Gemini support requires the 'google-genai' package. "
        "Install project dependencies with `uv sync`."
    ) from exc


OutputModel = TypeVar("OutputModel", bound=BaseModel)


class GeminiLLM:
    """Generate schema-validated output with Google's Gemini API.

    The API key is read from ``GEMINI_API_KEY`` by the Google GenAI SDK unless
    a configured client is passed in. ``model`` defaults to Gemini Flash for
    low-latency extraction workloads.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
        client: genai.Client | None = None,
        temperature: float = 0.0,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

        self.model = model
        self.temperature = temperature
        self.client = client or genai.Client(api_key=api_key)

    def generate_structured(
        self,
        prompt: str,
        *,
        response_schema: type[OutputModel],
        system_instruction: str | None = None,
    ) -> OutputModel:
        """Generate and validate a structured response."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        if isinstance(response.parsed, response_schema):
            return response.parsed
        if response.text is None:
            raise RuntimeError("Gemini returned no structured response")
        return response_schema.model_validate_json(response.text)

    @staticmethod
    def is_retryable_error(error: Exception) -> bool:
        """Return whether Gemini's response indicates a transient failure."""
        if isinstance(error, errors.ServerError):
            return True
        return isinstance(error, errors.ClientError) and error.code in (408, 429)

    def close(self) -> None:
        """Release the underlying HTTP client resources."""
        self.client.close()

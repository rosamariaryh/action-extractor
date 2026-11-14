"""Provider selection for configured LLM clients."""

from . import GeminiLLM
from .settings import LLMSettings
from observability.phoenix import setup_phoenix_tracing


def create_llm(settings: LLMSettings) -> GeminiLLM:
    """Create the LLM client selected by ``LLM_PROVIDER`` settings."""
    if settings.provider == "gemini":
        setup_phoenix_tracing()
        return GeminiLLM(model=settings.model, api_key=settings.api_key)
    raise ValueError(f"Unsupported LLM provider: {settings.provider}")

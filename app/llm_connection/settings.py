"""Application settings loaded from environment variables and the project .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str
    model: str
    max_concurrency: int
    max_retries: int
    requests_per_minute: int


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE .env entries without overriding process variables."""
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("export "):
            entry = entry[7:].lstrip()
        key, separator, value = entry.partition("=")
        if not separator or not key.strip():
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", maxsplit=1)[0].rstrip()
        os.environ.setdefault(key.strip(), value)


def load_llm_settings(env_file: Path | None = None) -> LLMSettings:
    """Load and validate provider and request-control settings."""
    project_root = Path(__file__).resolve().parents[2]
    _load_env_file(env_file or project_root / ".env")

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip()

    rpm_value = os.getenv("LLM_REQ_PER_MINUTE") or os.getenv("LLM_REQUESTS_PER_MINUTE", "60")

    try:
        max_concurrency = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        requests_per_minute = int(rpm_value)
    except ValueError as exc:
        raise ValueError("LLM concurrency, retry, and rate settings must be integers") from exc

    if provider != "gemini":
        raise ValueError(f"Unsupported LLM_PROVIDER {provider!r}; currently supported: gemini")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required (set it in .env or the process environment)")
    if not model:
        raise ValueError("GEMINI_MODEL_NAME must not be empty")
    if max_concurrency < 1 or requests_per_minute < 1 or max_retries < 0:
        raise ValueError(
            "LLM_MAX_CONCURRENCY and request rate must be positive; "
            "LLM_MAX_RETRIES must be zero or greater"
        )

    return LLMSettings(
        provider=provider,
        api_key=api_key,
        model=model,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        requests_per_minute=requests_per_minute,
    )

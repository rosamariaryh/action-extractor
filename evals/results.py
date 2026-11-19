"""Small result object shared by the project's evaluation functions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    score: float | None
    label: str
    explanation: str

"""Small result object shared by the project's evaluation functions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    label: str
    explanation: str

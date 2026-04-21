"""Abstract verifier interface. All domain verifiers implement this."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .task_schema import TaskSpec


@dataclass
class VerificationResult:
    passed: bool
    overall_score: float                     # 0.0 – 1.0
    dimension_scores: dict[str, float]       # e.g. {"structural": 0.9, "cost": 0.7}
    gaming_detected: bool = False
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        assert 0.0 <= self.overall_score <= 1.0


class BaseVerifier(ABC):
    """
    Verifies simulation results against a task spec.

    Basic subclass: simple pass/fail, open source, can be gamed.
    Pro subclass:   multi-dimensional, anti-gaming, calls paid API.
    """

    @abstractmethod
    def verify(self, sim_result: object, task: TaskSpec) -> VerificationResult:
        """Score sim_result against task requirements. Return VerificationResult."""

    def _weighted_score(self, scores: dict[str, float], weights: dict[str, float]) -> float:
        total_weight = sum(weights.values())
        return sum(scores[k] * weights[k] for k in scores if k in weights) / total_weight

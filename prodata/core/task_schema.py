"""Shared Pydantic schema for all Prodata task JSON files."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class GradingDimension(BaseModel):
    weight: float = Field(ge=0.0, le=1.0)
    checks: dict[str, Any] = {}


class GradingCriteria(BaseModel):
    weights: dict[str, float]             # e.g. {"structural": 0.35, "cost": 0.20}
    dimensions: dict[str, GradingDimension] = {}


class TaskRequirements(BaseModel):
    """Domain-agnostic requirements wrapper. Domain-specific fields go in 'extra'."""
    model_config = {"extra": "allow"}     # Allow domain-specific fields


class TaskSpec(BaseModel):
    task_id: str
    domain: str                           # "mechanical" | "rf" | "solar"
    category: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    requirements: TaskRequirements
    grading_criteria: GradingCriteria
    tags: list[str] = []
    version: str = "1.0"

    # Optional reference solution metadata
    reference_solution_exists: bool = False
    expected_score: float | None = None

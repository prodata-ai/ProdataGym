"""Domain-specific Pydantic schemas for CAD gym tasks."""

from pydantic import Field
from prodata.core.task_schema import TaskRequirements, TaskSpec


class BracketRequirements(TaskRequirements):
    """Typed requirements for bracket design tasks."""
    load_kg: float
    extension_mm: float
    material: str = "aluminum_6061_t6"
    yield_strength_mpa: float = 276.0
    process: str = "cnc_milling"
    max_cost_usd: float = 50.0
    max_deflection_mm: float = 5.0
    max_bounding_box_mm: list[float] = Field(default=[300.0, 300.0, 50.0])
    lateral_load_kg: float = 0.0
    temp_range_c: list[float] = Field(default=[-20.0, 70.0])


class BracketTaskSpec(TaskSpec):
    """TaskSpec with typed BracketRequirements."""
    requirements: BracketRequirements

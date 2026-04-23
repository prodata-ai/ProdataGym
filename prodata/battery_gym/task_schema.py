"""Domain-specific Pydantic schemas for battery gym tasks."""

from typing import Literal
from pydantic import Field
from prodata.core.task_schema import TaskRequirements, TaskSpec


class CellRequirements(TaskRequirements):
    """Typed requirements for battery cell design tasks."""

    # Chemistry choice drives base electrochemistry
    chemistry: Literal["NMC532", "LFP", "NCA"] = "NMC532"

    # Primary performance targets
    target_energy_density_whkg: float          # Wh/kg — gravimetric energy density
    min_cycle_life: int                         # cycles to 80% capacity retention

    # Operating constraints
    max_peak_temp_c: float = 50.0              # °C max cell temperature during operation
    max_cost_kwh: float = 150.0               # USD/kWh estimated cell cost

    # Duty cycle
    c_rate_charge: float = 1.0                 # C-rate for charging (1C = full charge in 1 hour)
    c_rate_discharge: float = 1.0              # C-rate for discharging

    # Environmental
    ambient_temp_c: float = 25.0              # °C ambient temperature during operation


class CellTaskSpec(TaskSpec):
    """TaskSpec with typed CellRequirements."""

    requirements: CellRequirements

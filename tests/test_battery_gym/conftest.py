"""Shared fixtures for battery gym tests."""

import pytest

from prodata.core.base_simulator import SimulationResult
from prodata.core.base_verifier import VerificationResult
from prodata.battery_gym.task_schema import CellRequirements, CellTaskSpec
from prodata.core.task_schema import GradingCriteria


@pytest.fixture
def standard_nmc_task() -> CellTaskSpec:
    return CellTaskSpec(
        task_id="test_ev_standard_001",
        domain="battery",
        category="ev_traction",
        difficulty="easy",
        description="Test NMC cell — 250 Wh/kg, 800 cycles, 45°C, 1C.",
        requirements=CellRequirements(
            chemistry="NMC532",
            target_energy_density_whkg=250.0,
            min_cycle_life=800,
            max_peak_temp_c=45.0,
            max_cost_kwh=100.0,
            c_rate_charge=1.0,
            c_rate_discharge=1.0,
            ambient_temp_c=25.0,
        ),
        grading_criteria=GradingCriteria(
            weights={"energy": 0.30, "cycle_life": 0.35, "thermal": 0.20, "cost": 0.15}
        ),
    )


@pytest.fixture
def lfp_task() -> CellTaskSpec:
    return CellTaskSpec(
        task_id="test_grid_lfp_001",
        domain="battery",
        category="grid_storage",
        difficulty="easy",
        description="Test LFP cell — 140 Wh/kg, 2000 cycles.",
        requirements=CellRequirements(
            chemistry="LFP",
            target_energy_density_whkg=140.0,
            min_cycle_life=2000,
            max_peak_temp_c=55.0,
            max_cost_kwh=80.0,
            c_rate_charge=0.5,
            c_rate_discharge=0.5,
            ambient_temp_c=25.0,
        ),
        grading_criteria=GradingCriteria(
            weights={"energy": 0.25, "cycle_life": 0.40, "thermal": 0.20, "cost": 0.15}
        ),
    )


@pytest.fixture
def passing_sim_result() -> SimulationResult:
    """Simulation result that should pass standard NMC task."""
    return SimulationResult(
        success=True,
        outputs={
            "energy_density_whkg": 265.0,
            "capacity_ah": 4.8,
            "peak_temperature_c": 38.0,
            "cycle_life_80pct": 950,
            "estimated_cost_kwh": 88.0,
            "discharge_curve": {},
            "param_warnings": [],
        },
    )


@pytest.fixture
def failing_energy_result() -> SimulationResult:
    """Simulation result with poor energy density."""
    return SimulationResult(
        success=True,
        outputs={
            "energy_density_whkg": 120.0,     # well below 250 target
            "capacity_ah": 2.2,
            "peak_temperature_c": 38.0,
            "cycle_life_80pct": 1000,
            "estimated_cost_kwh": 88.0,
            "discharge_curve": {},
            "param_warnings": [],
        },
    )


@pytest.fixture
def failing_thermal_result() -> SimulationResult:
    """Simulation result with thermal failure."""
    return SimulationResult(
        success=True,
        outputs={
            "energy_density_whkg": 255.0,
            "capacity_ah": 4.5,
            "peak_temperature_c": 62.0,     # exceeds 45°C limit by 17°C
            "cycle_life_80pct": 900,
            "estimated_cost_kwh": 88.0,
            "discharge_curve": {},
            "param_warnings": [],
        },
    )


@pytest.fixture
def failed_simulation() -> SimulationResult:
    """Completely failed simulation (e.g., code error)."""
    return SimulationResult(
        success=False,
        error="Agent code failed: NameError: params is not defined",
    )

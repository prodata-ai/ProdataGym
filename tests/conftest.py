"""Shared pytest fixtures."""

import pytest
from prodata.core.base_simulator import SimulationResult
from prodata.cad_gym.task_schema import BracketRequirements, BracketTaskSpec
from prodata.core.task_schema import GradingCriteria


def _make_task(
    task_id="test_001",
    load_kg=10.0,
    extension_mm=150.0,
    material="aluminum_6061_t6",
    yield_strength_mpa=276.0,
    process="cnc_milling",
    max_cost_usd=50.0,
    max_deflection_mm=3.0,
    max_bounding_box_mm=None,
    lateral_load_kg=0.0,
) -> BracketTaskSpec:
    return BracketTaskSpec(
        task_id=task_id,
        domain="mechanical",
        category="bracket",
        difficulty="medium",
        description="Test bracket task",
        requirements=BracketRequirements(
            load_kg=load_kg,
            extension_mm=extension_mm,
            material=material,
            yield_strength_mpa=yield_strength_mpa,
            process=process,
            max_cost_usd=max_cost_usd,
            max_deflection_mm=max_deflection_mm,
            max_bounding_box_mm=max_bounding_box_mm or [300.0, 300.0, 50.0],
            lateral_load_kg=lateral_load_kg,
        ),
        grading_criteria=GradingCriteria(
            weights={"structural": 0.50, "cost": 0.30, "geometry": 0.20}
        ),
    )


@pytest.fixture
def bracket_task() -> BracketTaskSpec:
    return _make_task()


@pytest.fixture
def passing_sim_result() -> SimulationResult:
    """A sim result that should comfortably pass all checks."""
    return SimulationResult(
        success=True,
        mesh_file="/tmp/test.stl",
        outputs={
            "volume_mm3": 50_000.0,
            "mass_kg": 0.135,
            "bounding_box_mm": [150.0, 100.0, 20.0],
            "max_stress_mpa": 60.0,
            "safety_factor": 4.6,
            "max_deflection_mm": 0.8,
            "material_cost_usd": 0.47,
            "fabrication_cost_usd": 50.0,
            "total_cost_usd": 40.0,
        },
    )


@pytest.fixture
def failing_structural_result() -> SimulationResult:
    """A sim result that fails structurally (safety factor < 1)."""
    return SimulationResult(
        success=True,
        mesh_file="/tmp/test.stl",
        outputs={
            "volume_mm3": 5_000.0,
            "mass_kg": 0.013,
            "bounding_box_mm": [150.0, 100.0, 2.0],
            "max_stress_mpa": 450.0,
            "safety_factor": 0.6,
            "max_deflection_mm": 12.0,
            "material_cost_usd": 0.05,
            "fabrication_cost_usd": 50.0,
            "total_cost_usd": 50.05,
        },
    )


@pytest.fixture
def failing_cost_result() -> SimulationResult:
    """A sim result that passes structurally but exceeds budget."""
    return SimulationResult(
        success=True,
        mesh_file="/tmp/test.stl",
        outputs={
            "volume_mm3": 500_000.0,
            "mass_kg": 1.35,
            "bounding_box_mm": [150.0, 100.0, 20.0],
            "max_stress_mpa": 20.0,
            "safety_factor": 13.8,
            "max_deflection_mm": 0.1,
            "material_cost_usd": 4.73,
            "fabrication_cost_usd": 50.0,
            "total_cost_usd": 200.0,
        },
    )


@pytest.fixture
def failed_simulation() -> SimulationResult:
    """A sim result where code execution itself failed."""
    return SimulationResult(success=False, error="SyntaxError in agent code")

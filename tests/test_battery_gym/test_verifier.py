"""Tests for BasicCellVerifier."""

import pytest
from prodata.battery_gym.verifiers.basic.cell_verifier import BasicCellVerifier
from prodata.core.base_simulator import SimulationResult


@pytest.fixture
def verifier() -> BasicCellVerifier:
    return BasicCellVerifier()


class TestVerifierFailedSimulation:
    def test_failed_sim_returns_zero_score(self, verifier, failed_simulation, standard_nmc_task):
        result = verifier.verify(failed_simulation, standard_nmc_task)
        assert result.passed is False
        assert result.overall_score == 0.0

    def test_failed_sim_all_dimensions_zero(self, verifier, failed_simulation, standard_nmc_task):
        result = verifier.verify(failed_simulation, standard_nmc_task)
        for v in result.dimension_scores.values():
            assert v == 0.0

    def test_failed_sim_includes_error_in_warnings(self, verifier, failed_simulation, standard_nmc_task):
        result = verifier.verify(failed_simulation, standard_nmc_task)
        assert len(result.warnings) > 0


class TestVerifierPassingDesign:
    def test_passing_result_scores_above_threshold(self, verifier, passing_sim_result, standard_nmc_task):
        result = verifier.verify(passing_sim_result, standard_nmc_task)
        assert result.overall_score >= 0.70

    def test_passing_result_passes(self, verifier, passing_sim_result, standard_nmc_task):
        result = verifier.verify(passing_sim_result, standard_nmc_task)
        assert result.passed is True

    def test_dimension_scores_all_present(self, verifier, passing_sim_result, standard_nmc_task):
        result = verifier.verify(passing_sim_result, standard_nmc_task)
        assert set(result.dimension_scores.keys()) == {"energy", "cycle_life", "thermal", "cost"}

    def test_dimension_scores_in_range(self, verifier, passing_sim_result, standard_nmc_task):
        result = verifier.verify(passing_sim_result, standard_nmc_task)
        for v in result.dimension_scores.values():
            assert 0.0 <= v <= 1.0

    def test_gaming_not_detected_in_basic(self, verifier, passing_sim_result, standard_nmc_task):
        result = verifier.verify(passing_sim_result, standard_nmc_task)
        assert result.gaming_detected is False


class TestVerifierEnergyScoring:
    def test_poor_energy_density_fails(self, verifier, failing_energy_result, standard_nmc_task):
        result = verifier.verify(failing_energy_result, standard_nmc_task)
        assert result.passed is False

    def test_poor_energy_gives_low_energy_score(self, verifier, failing_energy_result, standard_nmc_task):
        result = verifier.verify(failing_energy_result, standard_nmc_task)
        assert result.dimension_scores["energy"] < 0.3

    def test_energy_at_target_scores_high(self, verifier, standard_nmc_task):
        result_exact = SimulationResult(
            success=True,
            outputs={
                "energy_density_whkg": 250.0,  # exactly at target
                "capacity_ah": 4.5,
                "peak_temperature_c": 30.0,
                "cycle_life_80pct": 900,
                "estimated_cost_kwh": 85.0,
                "discharge_curve": {},
                "param_warnings": [],
            },
        )
        result = verifier.verify(result_exact, standard_nmc_task)
        assert result.dimension_scores["energy"] >= 0.95


class TestVerifierThermalScoring:
    def test_thermal_failure_lowers_score(self, verifier, failing_thermal_result, standard_nmc_task):
        result = verifier.verify(failing_thermal_result, standard_nmc_task)
        assert result.dimension_scores["thermal"] < 0.5

    def test_thermal_far_below_limit_scores_one(self, verifier, standard_nmc_task):
        cool_result = SimulationResult(
            success=True,
            outputs={
                "energy_density_whkg": 260.0,
                "capacity_ah": 4.6,
                "peak_temperature_c": 28.0,   # well below 45°C limit
                "cycle_life_80pct": 900,
                "estimated_cost_kwh": 88.0,
                "discharge_curve": {},
                "param_warnings": [],
            },
        )
        result = verifier.verify(cool_result, standard_nmc_task)
        assert result.dimension_scores["thermal"] == 1.0

    def test_thermal_limit_exceeded_triggers_warning(self, verifier, failing_thermal_result, standard_nmc_task):
        result = verifier.verify(failing_thermal_result, standard_nmc_task)
        assert any("temperature" in w.lower() or "temp" in w.lower() for w in result.warnings)


class TestVerifierCycleLife:
    def test_zero_cycle_life_fails(self, verifier, standard_nmc_task):
        dead_result = SimulationResult(
            success=True,
            outputs={
                "energy_density_whkg": 260.0,
                "capacity_ah": 4.5,
                "peak_temperature_c": 35.0,
                "cycle_life_80pct": 0,
                "estimated_cost_kwh": 88.0,
                "discharge_curve": {},
                "param_warnings": [],
            },
        )
        result = verifier.verify(dead_result, standard_nmc_task)
        assert result.passed is False
        assert result.dimension_scores["cycle_life"] < 0.1

    def test_cycle_life_above_target_scores_well(self, verifier, standard_nmc_task):
        long_life = SimulationResult(
            success=True,
            outputs={
                "energy_density_whkg": 255.0,
                "capacity_ah": 4.5,
                "peak_temperature_c": 35.0,
                "cycle_life_80pct": 1200,   # 50% above 800 target
                "estimated_cost_kwh": 88.0,
                "discharge_curve": {},
                "param_warnings": [],
            },
        )
        result = verifier.verify(long_life, standard_nmc_task)
        assert result.dimension_scores["cycle_life"] >= 0.90


class TestVerifierLFP:
    def test_lfp_task_verifies_correctly(self, verifier, lfp_task):
        lfp_result = SimulationResult(
            success=True,
            outputs={
                "energy_density_whkg": 148.0,
                "capacity_ah": 2.8,
                "peak_temperature_c": 33.0,
                "cycle_life_80pct": 3500,
                "estimated_cost_kwh": 65.0,
                "discharge_curve": {},
                "param_warnings": [],
            },
        )
        result = verifier.verify(lfp_result, lfp_task)
        assert result.passed is True
        assert result.overall_score >= 0.70


class TestVerifierCostScoring:
    def test_over_budget_lowers_cost_score(self, verifier, standard_nmc_task):
        expensive_result = SimulationResult(
            success=True,
            outputs={
                "energy_density_whkg": 255.0,
                "capacity_ah": 4.5,
                "peak_temperature_c": 35.0,
                "cycle_life_80pct": 900,
                "estimated_cost_kwh": 200.0,   # double the $100 budget
                "discharge_curve": {},
                "param_warnings": [],
            },
        )
        result = verifier.verify(expensive_result, standard_nmc_task)
        assert result.dimension_scores["cost"] == 0.0

    def test_at_budget_scores_one(self, verifier, standard_nmc_task):
        at_budget = SimulationResult(
            success=True,
            outputs={
                "energy_density_whkg": 255.0,
                "capacity_ah": 4.5,
                "peak_temperature_c": 35.0,
                "cycle_life_80pct": 900,
                "estimated_cost_kwh": 99.0,   # just under $100/kWh budget
                "discharge_curve": {},
                "param_warnings": [],
            },
        )
        result = verifier.verify(at_budget, standard_nmc_task)
        assert result.dimension_scores["cost"] == 1.0

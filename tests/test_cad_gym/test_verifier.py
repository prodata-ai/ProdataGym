"""
Unit tests for BasicBracketVerifier.

No CadQuery or trimesh required — SimulationResult is constructed directly
so the verifier logic is tested in isolation.
"""

import pytest
from prodata.cad_gym.verifiers.basic.bracket_verifier import BasicBracketVerifier
from prodata.core.base_verifier import VerificationResult
from tests.conftest import _make_task


class TestBasicBracketVerifier:
    @pytest.fixture(autouse=True)
    def verifier(self):
        self.v = BasicBracketVerifier()

    # ------------------------------------------------------------------
    # Passing designs
    # ------------------------------------------------------------------

    def test_passing_design_returns_passed_true(self, bracket_task, passing_sim_result):
        result = self.v.verify(passing_sim_result, bracket_task)
        assert result.passed is True

    def test_passing_design_score_between_zero_and_one(self, bracket_task, passing_sim_result):
        result = self.v.verify(passing_sim_result, bracket_task)
        assert 0.0 <= result.overall_score <= 1.0

    def test_passing_design_has_all_dimension_scores(self, bracket_task, passing_sim_result):
        result = self.v.verify(passing_sim_result, bracket_task)
        assert "structural" in result.dimension_scores
        assert "cost" in result.dimension_scores
        assert "geometry" in result.dimension_scores

    def test_passing_design_no_gaming_flagged(self, bracket_task, passing_sim_result):
        result = self.v.verify(passing_sim_result, bracket_task)
        assert result.gaming_detected is False

    # ------------------------------------------------------------------
    # Structural failure
    # ------------------------------------------------------------------

    def test_low_safety_factor_fails(self, bracket_task, failing_structural_result):
        result = self.v.verify(failing_structural_result, bracket_task)
        assert result.passed is False

    def test_low_safety_factor_low_structural_score(self, bracket_task, failing_structural_result):
        result = self.v.verify(failing_structural_result, bracket_task)
        assert result.dimension_scores["structural"] < 0.5

    def test_low_safety_factor_triggers_warning(self, bracket_task, failing_structural_result):
        result = self.v.verify(failing_structural_result, bracket_task)
        assert any("safety factor" in w.lower() for w in result.warnings)

    def test_exceeding_deflection_limit_penalises_structural(self, bracket_task):
        from prodata.core.base_simulator import SimulationResult
        sim = SimulationResult(
            success=True,
            outputs={
                "safety_factor": 3.0,
                "max_deflection_mm": 20.0,  # way over 3.0 mm limit
                "total_cost_usd": 30.0,
                "bounding_box_mm": [100.0, 80.0, 10.0],
            },
        )
        result = self.v.verify(sim, bracket_task)
        assert result.dimension_scores["structural"] < 0.8

    # ------------------------------------------------------------------
    # Cost failure
    # ------------------------------------------------------------------

    def test_over_budget_lowers_cost_score(self, bracket_task, failing_cost_result):
        result = self.v.verify(failing_cost_result, bracket_task)
        assert result.dimension_scores["cost"] < 0.5

    def test_over_budget_triggers_warning(self, bracket_task, failing_cost_result):
        result = self.v.verify(failing_cost_result, bracket_task)
        assert any("budget" in w.lower() for w in result.warnings)

    def test_exactly_at_budget_gets_full_cost_score(self, bracket_task):
        from prodata.core.base_simulator import SimulationResult
        sim = SimulationResult(
            success=True,
            outputs={
                "safety_factor": 4.0,
                "max_deflection_mm": 1.0,
                "total_cost_usd": 50.0,  # exactly at max_cost_usd=50
                "bounding_box_mm": [100.0, 80.0, 10.0],
            },
        )
        result = self.v.verify(sim, bracket_task)
        assert result.dimension_scores["cost"] == pytest.approx(1.0)

    # ------------------------------------------------------------------
    # Geometry checks
    # ------------------------------------------------------------------

    def test_oversized_bounding_box_fails_geometry(self, bracket_task):
        from prodata.core.base_simulator import SimulationResult
        sim = SimulationResult(
            success=True,
            outputs={
                "safety_factor": 4.0,
                "max_deflection_mm": 1.0,
                "total_cost_usd": 30.0,
                "bounding_box_mm": [9999.0, 9999.0, 9999.0],  # way over limit
            },
        )
        result = self.v.verify(sim, bracket_task)
        assert result.dimension_scores["geometry"] == 0.0

    def test_fitting_bounding_box_passes_geometry(self, bracket_task):
        from prodata.core.base_simulator import SimulationResult
        sim = SimulationResult(
            success=True,
            outputs={
                "safety_factor": 4.0,
                "max_deflection_mm": 1.0,
                "total_cost_usd": 30.0,
                "bounding_box_mm": [100.0, 80.0, 10.0],  # fits in [300, 300, 50]
            },
        )
        result = self.v.verify(sim, bracket_task)
        assert result.dimension_scores["geometry"] == 1.0

    # ------------------------------------------------------------------
    # Simulation failure
    # ------------------------------------------------------------------

    def test_failed_simulation_returns_score_zero(self, bracket_task, failed_simulation):
        result = self.v.verify(failed_simulation, bracket_task)
        assert result.overall_score == 0.0
        assert result.passed is False

    def test_failed_simulation_propagates_error_message(self, bracket_task, failed_simulation):
        result = self.v.verify(failed_simulation, bracket_task)
        assert len(result.warnings) > 0

    # ------------------------------------------------------------------
    # Score properties
    # ------------------------------------------------------------------

    def test_overall_score_is_weighted_average(self, bracket_task, passing_sim_result):
        result = self.v.verify(passing_sim_result, bracket_task)
        # Verify composite score is in expected range given dimension scores
        structural = result.dimension_scores["structural"]
        cost = result.dimension_scores["cost"]
        geometry = result.dimension_scores["geometry"]
        expected = structural * 0.50 + cost * 0.30 + geometry * 0.20
        assert result.overall_score == pytest.approx(expected, abs=1e-4)

    def test_all_dimension_scores_in_valid_range(self, bracket_task, passing_sim_result):
        result = self.v.verify(passing_sim_result, bracket_task)
        for dim, score in result.dimension_scores.items():
            assert 0.0 <= score <= 1.0, f"Dimension '{dim}' score {score} out of range"

    def test_high_safety_factor_gives_high_structural_score(self):
        from prodata.core.base_simulator import SimulationResult
        task = _make_task(max_deflection_mm=5.0)
        sim = SimulationResult(
            success=True,
            outputs={
                "safety_factor": 5.0,
                "max_deflection_mm": 0.5,
                "total_cost_usd": 25.0,
                "bounding_box_mm": [100.0, 80.0, 15.0],
            },
        )
        result = BasicBracketVerifier().verify(sim, task)
        assert result.dimension_scores["structural"] > 0.9

    def test_tight_deflection_requirement_penalises_borderline_design(self):
        from prodata.core.base_simulator import SimulationResult
        task = _make_task(max_deflection_mm=0.1)  # very tight
        sim = SimulationResult(
            success=True,
            outputs={
                "safety_factor": 3.5,
                "max_deflection_mm": 0.5,  # over the 0.1 mm limit
                "total_cost_usd": 25.0,
                "bounding_box_mm": [100.0, 80.0, 15.0],
            },
        )
        result = BasicBracketVerifier().verify(sim, task)
        assert result.dimension_scores["structural"] < 0.8

    # ------------------------------------------------------------------
    # Material variety — verifier is material-agnostic (uses sim outputs)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("material", [
        "aluminum_6061_t6", "aluminum_7075_t6", "steel_mild",
        "steel_316", "steel_4140", "titanium_6al4v", "pla",
    ])
    def test_verifier_works_for_all_materials(self, material):
        from prodata.core.base_simulator import SimulationResult
        task = _make_task(material=material)
        sim = SimulationResult(
            success=True,
            outputs={
                "safety_factor": 3.0,
                "max_deflection_mm": 1.0,
                "total_cost_usd": 30.0,
                "bounding_box_mm": [120.0, 100.0, 15.0],
            },
        )
        result = BasicBracketVerifier().verify(sim, task)
        assert 0.0 <= result.overall_score <= 1.0

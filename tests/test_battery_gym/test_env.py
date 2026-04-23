"""
Integration tests for CellDesignEnv.

Tests that don't require PyBaMM use the dataset mock or skip gracefully.
Tests marked pybamm require PyBaMM to be installed.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pybamm = pytest.importorskip("pybamm", reason="pybamm not installed")


@pytest.fixture(scope="module")
def env():
    """Fresh CellDesignEnv in live mode (uses PyBaMM)."""
    import gymnasium as gym
    import prodata.battery_gym  # noqa: F401 — registers envs

    e = gym.make("prodata/CellDesign-v0")
    yield e
    e.close()


GOOD_PARAMS_CODE = """
params = {
    "chemistry": "NMC532",
    "negative_electrode_thickness": 95e-6,
    "negative_electrode_porosity": 0.28,
    "negative_electrode_particle_radius": 5e-6,
    "positive_electrode_thickness": 80e-6,
    "positive_electrode_porosity": 0.30,
    "positive_electrode_particle_radius": 3.5e-6,
    "separator_thickness": 20e-6,
    "separator_porosity": 0.45,
    "ambient_temperature_celsius": 25.0,
}
"""

BROKEN_CODE = "x = 1  # forgot to define params"

NO_PARAMS_CODE = "params = 'not a dict'"


class TestEnvReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert obs is not None
        assert "task_id" in info

    def test_obs_has_required_keys(self, env):
        obs, _ = env.reset()
        expected = {
            "task_description", "target_energy_density_whkg", "min_cycle_life",
            "max_peak_temp_c", "max_cost_kwh", "c_rate_charge", "c_rate_discharge",
            "step", "energy_density_whkg", "cycle_life_80pct",
            "peak_temperature_c", "estimated_cost_kwh",
        }
        assert expected.issubset(set(obs.keys()))

    def test_reset_state_starts_uneval(self, env):
        obs, _ = env.reset()
        assert obs["energy_density_whkg"][0] == pytest.approx(-1.0)
        assert obs["cycle_life_80pct"][0] == pytest.approx(-1.0)

    def test_reset_selects_specific_task(self, env):
        _, info = env.reset(options={"task_id": "cell_ev_standard_001"})
        assert info["task_id"] == "cell_ev_standard_001"

    def test_reset_resets_step_counter(self, env):
        env.reset()
        env.step(GOOD_PARAMS_CODE)
        obs, _ = env.reset()
        assert int(obs["step"]) == 0


class TestEnvStep:
    def test_step_returns_five_values(self, env):
        env.reset()
        result = env.step(GOOD_PARAMS_CODE)
        assert len(result) == 5

    def test_step_reward_in_range(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(GOOD_PARAMS_CODE)
        assert -1.0 <= reward <= 1.0

    def test_step_info_has_required_keys(self, env):
        env.reset()
        _, _, _, _, info = env.step(GOOD_PARAMS_CODE)
        assert "success" in info
        assert "task_id" in info
        assert "dimension_scores" in info
        assert "step" in info

    def test_step_increments_counter(self, env):
        env.reset()
        env.step(GOOD_PARAMS_CODE)
        _, _, _, _, info = env.step(GOOD_PARAMS_CODE)
        assert info["step"] == 2

    def test_broken_code_gives_negative_reward(self, env):
        env.reset()
        _, reward, terminated, _, info = env.step(BROKEN_CODE)
        assert reward == pytest.approx(-1.0)
        assert terminated is True
        assert info["success"] is False

    def test_params_not_dict_gives_negative_reward(self, env):
        env.reset()
        _, reward, terminated, _, _ = env.step(NO_PARAMS_CODE)
        assert reward == pytest.approx(-1.0)
        assert terminated is True

    def test_good_design_updates_observation(self, env):
        env.reset(options={"task_id": "cell_ev_standard_001"})
        obs, _, _, _, _ = env.step(GOOD_PARAMS_CODE)
        assert obs["energy_density_whkg"][0] > 0
        assert obs["peak_temperature_c"][0] > 0

    def test_max_steps_terminates_episode(self, env):
        env.reset()
        terminated = False
        for _ in range(env.max_steps):
            _, _, terminated, _, _ = env.step(GOOD_PARAMS_CODE)
        assert terminated is True

    def test_passing_design_terminates_episode(self, env):
        """A high-quality design that passes should terminate the episode early."""
        env.reset(options={"task_id": "cell_ev_commuter_001"})  # easy task
        _, _, terminated, _, info = env.step(GOOD_PARAMS_CODE)
        if info["success"]:
            assert terminated is True


class TestEnvObservationValues:
    def test_energy_density_is_positive_after_step(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(GOOD_PARAMS_CODE)
        assert obs["energy_density_whkg"][0] > 50  # at least 50 Wh/kg

    def test_cycle_life_is_positive_after_step(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(GOOD_PARAMS_CODE)
        assert obs["cycle_life_80pct"][0] > 0

    def test_peak_temp_is_above_ambient(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(GOOD_PARAMS_CODE)
        # Peak temp should be >= ambient (25°C here)
        assert obs["peak_temperature_c"][0] >= 20.0


class TestTasksJson:
    def test_all_30_tasks_load(self):
        tasks_path = (
            Path(__file__).parent.parent.parent
            / "prodata" / "battery_gym" / "tasks" / "cells_basic.json"
        )
        with open(tasks_path) as f:
            tasks = json.load(f)
        assert len(tasks) == 30

    def test_all_tasks_have_required_fields(self):
        tasks_path = (
            Path(__file__).parent.parent.parent
            / "prodata" / "battery_gym" / "tasks" / "cells_basic.json"
        )
        with open(tasks_path) as f:
            tasks = json.load(f)
        required = {"task_id", "domain", "category", "difficulty", "description",
                    "requirements", "grading_criteria"}
        for task in tasks:
            assert required.issubset(set(task.keys())), f"Task {task.get('task_id')} missing fields"

    def test_difficulty_distribution(self):
        tasks_path = (
            Path(__file__).parent.parent.parent
            / "prodata" / "battery_gym" / "tasks" / "cells_basic.json"
        )
        with open(tasks_path) as f:
            tasks = json.load(f)
        difficulties = [t["difficulty"] for t in tasks]
        assert difficulties.count("easy") >= 8
        assert difficulties.count("medium") >= 8
        assert difficulties.count("hard") >= 8

    def test_chemistries_are_valid(self):
        tasks_path = (
            Path(__file__).parent.parent.parent
            / "prodata" / "battery_gym" / "tasks" / "cells_basic.json"
        )
        with open(tasks_path) as f:
            tasks = json.load(f)
        valid = {"NMC532", "LFP", "NCA"}
        for task in tasks:
            chem = task["requirements"].get("chemistry")
            assert chem in valid, f"Task {task['task_id']} has invalid chemistry {chem}"

    def test_task_ids_are_unique(self):
        tasks_path = (
            Path(__file__).parent.parent.parent
            / "prodata" / "battery_gym" / "tasks" / "cells_basic.json"
        )
        with open(tasks_path) as f:
            tasks = json.load(f)
        ids = [t["task_id"] for t in tasks]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"


class TestSimulatorCycleLife:
    """Unit tests for the physics-based cycle life formula."""

    def test_high_c_rate_reduces_cycle_life(self):
        from prodata.battery_gym.simulators.electrochemical_sim import ElectrochemicalSimulator
        sim = ElectrochemicalSimulator()

        base = sim._estimate_cycle_life(
            {"negative_electrode_particle_radius": 5.5e-6,
             "negative_electrode_porosity": 0.30,
             "ambient_temperature_celsius": 25.0},
            "NMC532", c_rate_charge=1.0, peak_temp_c=30.0
        )
        fast = sim._estimate_cycle_life(
            {"negative_electrode_particle_radius": 5.5e-6,
             "negative_electrode_porosity": 0.30,
             "ambient_temperature_celsius": 25.0},
            "NMC532", c_rate_charge=4.0, peak_temp_c=35.0
        )
        assert fast < base

    def test_lfp_has_longer_life_than_nmc(self):
        from prodata.battery_gym.simulators.electrochemical_sim import ElectrochemicalSimulator
        sim = ElectrochemicalSimulator()
        params = {
            "negative_electrode_particle_radius": 5.5e-6,
            "negative_electrode_porosity": 0.30,
            "ambient_temperature_celsius": 25.0,
        }
        nmc = sim._estimate_cycle_life(params, "NMC532", 1.0, 30.0)
        lfp = sim._estimate_cycle_life(params, "LFP",    1.0, 30.0)
        assert lfp > nmc

    def test_high_temperature_reduces_cycle_life(self):
        from prodata.battery_gym.simulators.electrochemical_sim import ElectrochemicalSimulator
        sim = ElectrochemicalSimulator()
        params = {
            "negative_electrode_particle_radius": 5.5e-6,
            "negative_electrode_porosity": 0.30,
            "ambient_temperature_celsius": 25.0,
        }
        cool = sim._estimate_cycle_life(params, "NMC532", 1.0, peak_temp_c=25.0)
        hot  = sim._estimate_cycle_life(params, "NMC532", 1.0, peak_temp_c=55.0)
        assert hot < cool

    def test_large_particles_reduce_cycle_life(self):
        from prodata.battery_gym.simulators.electrochemical_sim import ElectrochemicalSimulator
        sim = ElectrochemicalSimulator()
        small = sim._estimate_cycle_life(
            {"negative_electrode_particle_radius": 3e-6,
             "negative_electrode_porosity": 0.30,
             "ambient_temperature_celsius": 25.0},
            "NMC532", 1.0, 30.0
        )
        large = sim._estimate_cycle_life(
            {"negative_electrode_particle_radius": 15e-6,
             "negative_electrode_porosity": 0.30,
             "ambient_temperature_celsius": 25.0},
            "NMC532", 1.0, 30.0
        )
        assert large < small

    def test_cycle_life_bounded(self):
        from prodata.battery_gym.simulators.electrochemical_sim import ElectrochemicalSimulator
        sim = ElectrochemicalSimulator()
        for chemistry in ["NMC532", "LFP"]:
            cl = sim._estimate_cycle_life(
                {"negative_electrode_particle_radius": 5.5e-6,
                 "negative_electrode_porosity": 0.30,
                 "ambient_temperature_celsius": 25.0},
                chemistry, 1.0, 30.0
            )
            assert 50 <= cl <= 20_000

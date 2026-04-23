"""CellDesign-v0: Agent designs battery cell parameters using PyBaMM verification."""

from pathlib import Path
from typing import Any

import string
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Printable ASCII + common symbols used in task descriptions.
# Superset of string.printable that is explicit about what we allow.
_TEXT_CHARSET = string.printable

from prodata.core import ProdataEnv, TaskSpec
from prodata.battery_gym.task_schema import CellRequirements, CellTaskSpec
from prodata.battery_gym.simulators.electrochemical_sim import ElectrochemicalSimulator
from prodata.battery_gym.verifiers.basic.cell_verifier import BasicCellVerifier

_DEFAULT_TASKS = Path(__file__).parent.parent / "tasks" / "cells_basic.json"


class CellDesignEnv(ProdataEnv):
    """
    Gymnasium environment for lithium-ion battery cell design.

    The agent generates Python code that defines `params` — a dict of
    electrode and separator design parameters. PyBaMM simulates the cell
    and the verifier scores it against the task requirements.

    Observation: dict with task specs + current simulation state
    Action:      Python code string that defines `params` dict
    Reward:      weighted verification score in [0, 1]

    Quick start:
        import gymnasium as gym
        import prodata.battery_gym

        env = gym.make("prodata/CellDesign-v0")
        obs, info = env.reset()
        print(obs["task_description"])

        action = '''
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
        '''
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Score: {reward:.2f} | Passed: {info['success']}")
        print(f"Energy: {info['dimension_scores']}")

    Simulator modes:
        mode="live"    — runs real PyBaMM per step (~2–5 s/step)
        mode="dataset" — KDTree lookup in pre-computed dataset (~1 ms/step)
                         requires dataset_path to a .parquet file
                         generate with: python -m prodata.battery_gym.scripts.precompute_dataset
    """

    action_space = spaces.Text(max_length=8000)

    def __init__(
        self,
        tasks_path: str | Path | None = None,
        verifier_mode: str = "basic",
        max_steps: int = 5,
        render_mode: str | None = None,
        mode: str = "live",
        dataset_path: str | Path | None = None,
    ):
        super().__init__(
            tasks_path=tasks_path or _DEFAULT_TASKS,
            verifier_mode=verifier_mode,
            max_steps=max_steps,
            render_mode=render_mode,
            task_class=CellTaskSpec,
        )

        self.simulator = ElectrochemicalSimulator(mode=mode, dataset_path=dataset_path)

        if verifier_mode == "pro":
            from prodata.battery_gym.verifiers.pro.cell_verifier_pro import ProCellVerifier
            self.verifier = ProCellVerifier()
        else:
            self.verifier = BasicCellVerifier()

        self.observation_space = spaces.Dict({
            "task_description":            spaces.Text(3000, charset=_TEXT_CHARSET),
            # Requirements
            "target_energy_density_whkg":  spaces.Box(0, 1000, shape=(1,), dtype=np.float32),
            "min_cycle_life":              spaces.Box(0, 20000, shape=(1,), dtype=np.float32),
            "max_peak_temp_c":             spaces.Box(-40, 150, shape=(1,), dtype=np.float32),
            "max_cost_kwh":                spaces.Box(0, 1000, shape=(1,), dtype=np.float32),
            "c_rate_charge":               spaces.Box(0, 10, shape=(1,), dtype=np.float32),
            "c_rate_discharge":            spaces.Box(0, 10, shape=(1,), dtype=np.float32),
            "step":                        spaces.Discrete(self.max_steps + 1),
            # Current design state (-1.0 = not yet evaluated)
            "energy_density_whkg":         spaces.Box(-1, 1000, shape=(1,), dtype=np.float32),
            "cycle_life_80pct":            spaces.Box(-1, 20000, shape=(1,), dtype=np.float32),
            "peak_temperature_c":          spaces.Box(-1, 200, shape=(1,), dtype=np.float32),
            "estimated_cost_kwh":          spaces.Box(-1, 1000, shape=(1,), dtype=np.float32),
        })

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Replace non-printable-ASCII characters for Gymnasium Text space compliance."""
        return (
            text
            .replace("\u00b0", "deg")   # ° → deg
            .replace("\u2013", "-")     # en-dash → hyphen
            .replace("\u2014", "--")    # em-dash → --
            .encode("ascii", errors="replace")
            .decode("ascii")
        )

    def _build_observation(self, task: CellTaskSpec, sim_result: Any) -> dict:
        req: CellRequirements = task.requirements
        obs = {
            "task_description":           self._sanitize_text(task.description),
            "target_energy_density_whkg": np.array([req.target_energy_density_whkg], dtype=np.float32),
            "min_cycle_life":             np.array([float(req.min_cycle_life)], dtype=np.float32),
            "max_peak_temp_c":            np.array([req.max_peak_temp_c], dtype=np.float32),
            "max_cost_kwh":               np.array([req.max_cost_kwh], dtype=np.float32),
            "c_rate_charge":              np.array([req.c_rate_charge], dtype=np.float32),
            "c_rate_discharge":           np.array([req.c_rate_discharge], dtype=np.float32),
            "step":                       np.int64(self._current_step),
            # Design state: -1.0 until first step
            "energy_density_whkg":        np.array([-1.0], dtype=np.float32),
            "cycle_life_80pct":           np.array([-1.0], dtype=np.float32),
            "peak_temperature_c":         np.array([-1.0], dtype=np.float32),
            "estimated_cost_kwh":         np.array([-1.0], dtype=np.float32),
        }
        if sim_result is not None:
            out = sim_result.outputs
            obs["energy_density_whkg"]  = np.array([out.get("energy_density_whkg",  -1.0)], dtype=np.float32)
            obs["cycle_life_80pct"]     = np.array([float(out.get("cycle_life_80pct", -1))], dtype=np.float32)
            obs["peak_temperature_c"]   = np.array([out.get("peak_temperature_c",   -1.0)], dtype=np.float32)
            obs["estimated_cost_kwh"]   = np.array([out.get("estimated_cost_kwh",   -1.0)], dtype=np.float32)
        return obs

    def _run_step(self, action: str, task: TaskSpec):
        sim_result = self.simulator.execute(action, task.requirements.model_dump())
        verification = self.verifier.verify(sim_result, task)
        return sim_result, verification

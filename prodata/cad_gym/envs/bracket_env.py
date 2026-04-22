"""BracketDesign-v0: Agent designs mechanical brackets in CadQuery."""

from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from prodata.core import ProdataEnv, TaskSpec
from prodata.cad_gym.task_schema import BracketRequirements, BracketTaskSpec
from prodata.cad_gym.simulators.mechanical_sim import MechanicalSimulator
from prodata.cad_gym.verifiers.basic.bracket_verifier import BasicBracketVerifier

_DEFAULT_TASKS = Path(__file__).parent.parent / "tasks" / "brackets_basic.json"


class BracketDesignEnv(ProdataEnv):
    """
    Gymnasium environment for mechanical bracket design.

    Observation: dict with task description + current simulation state
    Action:      Python string using CadQuery — must define variable `result`
    Reward:      composite verification score in [0, 1]

    Usage:
        import gymnasium as gym
        import prodata.cad_gym
        env = gym.make("prodata/BracketDesign-v0")
    """

    # Action space: text (unbounded string)
    # Gymnasium doesn't have a Text space in all versions; use Box as token ids
    # or just document that action is a raw Python string passed directly.
    action_space = spaces.Text(max_length=8000)

    def __init__(
        self,
        tasks_path: str | Path | None = None,
        verifier_mode: str = "basic",
        max_steps: int = 5,
        render_mode: str | None = None,
    ):
        super().__init__(
            tasks_path=tasks_path or _DEFAULT_TASKS,
            verifier_mode=verifier_mode,
            max_steps=max_steps,
            render_mode=render_mode,
            task_class=BracketTaskSpec,
        )
        self.simulator = MechanicalSimulator()

        if verifier_mode == "pro":
            from prodata.cad_gym.verifiers.pro.bracket_verifier_pro import ProBracketVerifier
            self.verifier = ProBracketVerifier()
        else:
            self.verifier = BasicBracketVerifier()

        # Observation space: flat dict (text descriptions + scalar state)
        self.observation_space = spaces.Dict({
            "task_description": spaces.Text(2000),
            "load_kg": spaces.Box(0, 1000, shape=(1,), dtype=np.float32),
            "extension_mm": spaces.Box(0, 5000, shape=(1,), dtype=np.float32),
            "max_cost_usd": spaces.Box(0, 10000, shape=(1,), dtype=np.float32),
            "step": spaces.Discrete(self.max_steps + 1),
            # Current design state (None on first step)
            "safety_factor": spaces.Box(-1, 100, shape=(1,), dtype=np.float32),
            "cost_usd": spaces.Box(-1, 10000, shape=(1,), dtype=np.float32),
        })

    def _build_observation(self, task: BracketTaskSpec, sim_result: Any) -> dict:
        req: BracketRequirements = task.requirements
        obs = {
            "task_description": task.description,
            "load_kg": np.array([req.load_kg], dtype=np.float32),
            "extension_mm": np.array([req.extension_mm], dtype=np.float32),
            "max_cost_usd": np.array([req.max_cost_usd], dtype=np.float32),
            "step": np.int64(self._current_step),
            "safety_factor": np.array([-1.0], dtype=np.float32),
            "cost_usd": np.array([-1.0], dtype=np.float32),
        }
        if sim_result is not None:
            obs["safety_factor"] = np.array(
                [sim_result.outputs.get("safety_factor", -1.0)], dtype=np.float32
            )
            obs["cost_usd"] = np.array(
                [sim_result.outputs.get("total_cost_usd", -1.0)], dtype=np.float32
            )
        return obs

    def _run_step(self, action: str, task: TaskSpec):
        sim_result = self.simulator.execute(action, task.requirements.model_dump())
        verification = self.verifier.verify(sim_result, task)
        return sim_result, verification

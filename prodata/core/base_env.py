"""Base Gymnasium environment all Prodata domain envs inherit from."""

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Type

import gymnasium as gym
import numpy as np

from .task_schema import TaskSpec


class ProdataEnv(gym.Env):
    """
    Base class for all Prodata RL environments.

    Subclasses must implement:
      - _build_observation(task, sim_result) → dict
      - _run_step(action, task) → (SimulationResult, VerificationResult)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        tasks_path: str | Path,
        verifier_mode: str = "basic",  # "basic" | "pro"
        max_steps: int = 5,
        render_mode: str | None = None,
        task_class: Type[TaskSpec] = TaskSpec,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.verifier_mode = verifier_mode

        self._tasks: list[TaskSpec] = self._load_tasks(tasks_path, task_class)
        self._current_task: TaskSpec | None = None
        self._current_step: int = 0
        self._last_sim_result: Any = None

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_observation(self, task: TaskSpec, sim_result: Any) -> dict:
        """Return observation dict for the current state."""

    @abstractmethod
    def _run_step(self, action: str, task: TaskSpec) -> tuple[Any, Any]:
        """Execute action. Return (SimulationResult, VerificationResult)."""

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if options and "task_id" in options:
            self._current_task = next(
                t for t in self._tasks if t.task_id == options["task_id"]
            )
        else:
            self._current_task = self.np_random.choice(self._tasks)

        self._current_step = 0
        self._last_sim_result = None

        obs = self._build_observation(self._current_task, None)
        info = {"task_id": self._current_task.task_id}
        return obs, info

    def step(self, action: str):
        assert self._current_task is not None, "Call reset() before step()"
        self._current_step += 1

        try:
            sim_result, verification = self._run_step(action, self._current_task)
        except Exception as exc:
            obs = self._build_observation(self._current_task, None)
            return obs, -1.0, True, False, {
                "error": str(exc),
                "success": False,
                "dimension_scores": {},
                "gaming_detected": False,
                "warnings": [str(exc)],
                "step": self._current_step,
            }

        self._last_sim_result = sim_result
        reward = verification.overall_score
        terminated = verification.passed or self._current_step >= self.max_steps

        obs = self._build_observation(self._current_task, sim_result)
        info = {
            "success": verification.passed,
            "dimension_scores": verification.dimension_scores,
            "gaming_detected": verification.gaming_detected,
            "warnings": verification.warnings,
            "step": self._current_step,
        }
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_tasks(self, path: str | Path, task_class: Type[TaskSpec] = TaskSpec) -> list[TaskSpec]:
        path = Path(path)
        with open(path) as f:
            raw = json.load(f)
        return [task_class(**t) for t in raw]

    def task_ids(self) -> list[str]:
        return [t.task_id for t in self._tasks]

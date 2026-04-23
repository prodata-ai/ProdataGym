"""
Battery Gym — RL environments for lithium-ion cell design.

Environments:
  prodata/CellDesign-v0  — design battery cell parameters to meet performance specs

Usage:
    import gymnasium as gym
    import prodata.battery_gym  # registers environments

    env = gym.make("prodata/CellDesign-v0")
"""

import gymnasium as gym

from .envs.cell_design_env import CellDesignEnv

gym.register(
    id="prodata/CellDesign-v0",
    entry_point="prodata.battery_gym.envs.cell_design_env:CellDesignEnv",
    kwargs={
        "tasks_path": None,
        "verifier_mode": "basic",
        "max_steps": 5,
        "mode": "live",
    },
)

__all__ = ["CellDesignEnv"]

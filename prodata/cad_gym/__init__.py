"""
prodata.cad_gym — RL environments for mechanical CAD design.

Registers Gymnasium environments on import:
  - prodata/BracketDesign-v0
"""

from gymnasium.envs.registration import register

register(
    id="prodata/BracketDesign-v0",
    entry_point="prodata.cad_gym.envs.bracket_env:BracketDesignEnv",
    max_episode_steps=5,
)

# Future environments registered here:
# register(id="prodata/EnclosureDesign-v0", ...)
# register(id="prodata/GearTrain-v0", ...)

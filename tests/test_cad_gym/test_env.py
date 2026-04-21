"""
Integration tests for BracketDesignEnv.

Requires cadquery and trimesh. Skip automatically if not installed.
Run with: pytest tests/test_cad_gym/test_env.py -v
"""

import pytest

cadquery = pytest.importorskip("cadquery", reason="cadquery not installed")
trimesh = pytest.importorskip("trimesh", reason="trimesh not installed")

import gymnasium as gym
import prodata.cad_gym  # registers environments

pytestmark = pytest.mark.integration

# Minimal CadQuery code that produces a valid Workplane named 'result'
VALID_CODE = """
import cadquery as cq
result = cq.Workplane("XY").box(120, 80, 15)
"""

# Code that runs but produces degenerate geometry (very thin)
THIN_CODE = """
import cadquery as cq
result = cq.Workplane("XY").box(200, 100, 1)
"""

# Code with a syntax error
SYNTAX_ERROR_CODE = """
import cadquery as cq
result = cq.Workplane("XY"  # missing closing paren
"""

# Code that doesn't define 'result'
MISSING_RESULT_CODE = """
import cadquery as cq
my_bracket = cq.Workplane("XY").box(100, 80, 10)
"""


@pytest.fixture(scope="module")
def env():
    e = gym.make("prodata/BracketDesign-v0")
    yield e
    e.close()


class TestEnvCreation:
    def test_env_makes_without_error(self):
        e = gym.make("prodata/BracketDesign-v0")
        assert e is not None
        e.close()

    def test_env_has_correct_id(self):
        e = gym.make("prodata/BracketDesign-v0")
        assert "BracketDesign" in str(type(e))
        e.close()


class TestEnvReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_obs_has_required_keys(self, env):
        obs, _ = env.reset(seed=0)
        assert "task_description" in obs
        assert "load_kg" in obs
        assert "extension_mm" in obs
        assert "max_cost_usd" in obs
        assert "step" in obs

    def test_reset_info_has_task_id(self, env):
        _, info = env.reset(seed=0)
        assert "task_id" in info

    def test_reset_with_specific_task_id(self, env):
        _, info = env.reset(options={"task_id": "cad_bracket_001"})
        assert info["task_id"] == "cad_bracket_001"

    def test_reset_step_counter_is_zero(self, env):
        obs, _ = env.reset(seed=0)
        assert obs["step"] == 0

    def test_reset_simulation_state_is_empty(self, env):
        obs, _ = env.reset(seed=0)
        assert obs["safety_factor"][0] == pytest.approx(-1.0)
        assert obs["cost_usd"][0] == pytest.approx(-1.0)


class TestEnvStep:
    def test_step_with_valid_code_returns_tuple(self, env):
        env.reset(seed=0)
        result = env.step(VALID_CODE)
        assert len(result) == 5  # obs, reward, terminated, truncated, info

    def test_step_reward_in_valid_range(self, env):
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(VALID_CODE)
        assert -1.0 <= reward <= 1.0

    def test_step_info_has_success_key(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(VALID_CODE)
        assert "success" in info

    def test_step_info_has_dimension_scores(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(VALID_CODE)
        assert "dimension_scores" in info
        scores = info["dimension_scores"]
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_step_with_syntax_error_returns_penalty(self, env):
        env.reset(seed=0)
        _, reward, terminated, _, info = env.step(SYNTAX_ERROR_CODE)
        assert reward == pytest.approx(-1.0)
        assert terminated is True
        assert "error" in info

    def test_step_with_missing_result_variable_returns_penalty(self, env):
        env.reset(seed=0)
        _, reward, terminated, _, info = env.step(MISSING_RESULT_CODE)
        assert reward == pytest.approx(-1.0)
        assert terminated is True

    def test_step_increments_step_counter(self, env):
        obs, _ = env.reset(seed=0)
        assert obs["step"] == 0
        obs, _, _, _, _ = env.step(VALID_CODE)
        assert obs["step"] == 1

    def test_observation_updates_after_valid_step(self, env):
        env.reset(options={"task_id": "cad_bracket_001"})
        obs, reward, _, _, _ = env.step(VALID_CODE)
        if reward > 0:  # only check if sim succeeded
            assert obs["safety_factor"][0] > -1.0
            assert obs["cost_usd"][0] > -1.0


class TestEnvEpisodeFlow:
    def test_full_episode_terminates(self, env):
        env.reset(seed=42)
        for _ in range(10):  # max_steps is 5, so this will definitely terminate
            _, _, terminated, truncated, _ = env.step(VALID_CODE)
            if terminated or truncated:
                break
        assert terminated or truncated

    def test_reset_after_terminated_episode(self, env):
        env.reset(seed=0)
        # Force termination
        for _ in range(6):
            _, _, terminated, truncated, _ = env.step(VALID_CODE)
            if terminated or truncated:
                break
        # Should be able to reset cleanly
        obs, info = env.reset(seed=1)
        assert obs["step"] == 0

    def test_task_ids_accessible(self, env):
        ids = env.unwrapped.task_ids()
        assert len(ids) == 50
        assert "cad_bracket_001" in ids
        assert "cad_bracket_050" in ids

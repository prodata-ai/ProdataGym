"""
CAD-Gym Quickstart

Demonstrates:
  1. Creating the BracketDesign-v0 environment
  2. Running a single episode with a hard-coded design
  3. Interpreting the reward and info dict

Run:
    pip install prodata[cad]
    python examples/cad/01_quickstart.py
"""

import gymnasium as gym
import prodata.cad_gym  # registers environments


def main():
    env = gym.make("prodata/BracketDesign-v0", render_mode=None)
    obs, info = env.reset(seed=42)

    print("=== Task ===")
    print(obs["task_description"])
    print(f"Load: {obs['load_kg'][0]} kg | Extension: {obs['extension_mm'][0]} mm")
    print(f"Budget: ${obs['max_cost_usd'][0]:.0f}")
    print()

    # A simple L-bracket design in CadQuery
    design_code = """
import cadquery as cq

# L-bracket: horizontal shelf + vertical back plate
horizontal = (
    cq.Workplane("XY")
    .box(120, 80, 12)
)
vertical = (
    cq.Workplane("XZ")
    .box(120, 10, 100)
    .translate((0, -35, 56))
)
result = horizontal.union(vertical)
"""

    obs, reward, terminated, truncated, info = env.step(design_code)

    print("=== Result ===")
    print(f"Reward (overall score): {reward:.3f}")
    print(f"Passed: {info['success']}")
    print(f"Dimension scores: {info['dimension_scores']}")
    if info["warnings"]:
        print(f"Warnings: {info['warnings']}")

    env.close()


if __name__ == "__main__":
    main()

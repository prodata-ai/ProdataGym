"""
Run the full CAD benchmark: evaluate a model on all 50 tasks.

Usage:
    python examples/cad/02_run_benchmark.py --model gpt-4o
    python examples/cad/02_run_benchmark.py --model qwen  # uses local Qwen2.5-Coder

Output: results/cad_benchmark_<model>_<date>.json
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import gymnasium as gym

import prodata.cad_gym

TASKS_PATH = Path(__file__).parent.parent.parent / "prodata/cad_gym/tasks/brackets_basic.json"


def get_model_completion(model_name: str, prompt: str) -> str:
    """Return design code from the model. Swap out for your model."""
    if model_name == "dummy":
        # Minimal valid code for testing the harness
        return """
import cadquery as cq
result = cq.Workplane("XY").box(120, 80, 10)
"""
    raise NotImplementedError(f"Integrate your model: {model_name}")


def run_benchmark(model_name: str, n_tasks: int | None = None) -> dict:
    env = gym.make("prodata/BracketDesign-v0")

    with open(TASKS_PATH) as f:
        all_tasks = json.load(f)

    tasks = all_tasks[:n_tasks] if n_tasks else all_tasks
    results = []

    for i, task in enumerate(tasks):
        obs, info = env.reset(options={"task_id": task["task_id"]})

        prompt = (
            f"Design a mechanical bracket using CadQuery Python.\n\n"
            f"Task: {obs['task_description']}\n\n"
            f"Requirements:\n"
            f"  - Load: {obs['load_kg'][0]} kg\n"
            f"  - Extension: {obs['extension_mm'][0]} mm\n"
            f"  - Budget: ${obs['max_cost_usd'][0]:.0f}\n\n"
            f"Your code must define a variable `result` of type cadquery.Workplane."
        )

        code = get_model_completion(model_name, prompt)

        _, reward, _, _, step_info = env.step(code)

        results.append({
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "passed": step_info["success"],
            "score": round(reward, 4),
            "dimension_scores": step_info["dimension_scores"],
        })

        print(f"[{i+1}/{len(tasks)}] {task['task_id']}: {reward:.3f} ({'PASS' if step_info['success'] else 'FAIL'})")

    env.close()

    passed = sum(r["passed"] for r in results)
    summary = {
        "model": model_name,
        "date": str(date.today()),
        "pass_rate": round(passed / len(results), 3),
        "mean_score": round(sum(r["score"] for r in results) / len(results), 3),
        "n_tasks": len(results),
        "per_task": results,
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dummy")
    parser.add_argument("--n-tasks", type=int, default=None)
    args = parser.parse_args()

    summary = run_benchmark(args.model, args.n_tasks)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"cad_benchmark_{args.model}_{date.today()}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\nPass rate: {summary['pass_rate']*100:.1f}% | Mean score: {summary['mean_score']:.3f}")
    print(f"Saved: {out_path}")

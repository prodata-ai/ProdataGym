"""
Run the full battery benchmark: evaluate a model on all 30 CellDesign tasks.

Usage:
    # Dry-run with a dummy model (no LLM required — tests the harness itself)
    python examples/battery/run_benchmark.py --model dummy

    # Easy tasks only, fast iteration
    python examples/battery/run_benchmark.py --model dummy --difficulty easy

    # Use dataset mode for fast reward (no live PyBaMM)
    python examples/battery/run_benchmark.py --model dummy \
        --dataset data/battery_dataset_10k.parquet

Output:
    benchmarks/results/<model>_<date>.json   — full results JSON
    (also printed: per-task pass/fail and final summary)

Leaderboard format is defined in benchmarks/battery_leaderboard.md.
To integrate your own model, implement `get_model_completion()` below.
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import gymnasium as gym

import prodata.battery_gym  # noqa: F401 — registers CellDesign-v0

TASKS_PATH = (
    Path(__file__).parent.parent.parent
    / "prodata/battery_gym/tasks/cells_basic.json"
)

# ── Prompt template ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an electrochemical engineer. Write Python code that defines battery "
    "cell parameters for a 21700 cylindrical cell.\n\n"
    "Rules:\n"
    "- Define a variable named `params` as a Python dict.\n"
    "- Required keys: chemistry, negative_electrode_thickness, "
    "negative_electrode_porosity, negative_electrode_particle_radius, "
    "positive_electrode_thickness, positive_electrode_porosity, "
    "positive_electrode_particle_radius, separator_thickness, "
    "separator_porosity, ambient_temperature_celsius\n"
    "- chemistry must be one of: \"NMC532\", \"LFP\", \"NCA\"\n"
    "- Thicknesses in metres (e.g. 80e-6 for 80 µm, valid range: 20–200 µm)\n"
    "- Porosities as fractions 0.0–1.0 (typical: 0.20–0.55)\n"
    "- Particle radii in metres (e.g. 5e-6 for 5 µm, typical: 1–15 µm)\n"
    "- No imports, function definitions, or other output needed.\n"
)

def build_prompt(obs: dict) -> str:
    req_lines = [
        f"  Target energy density:  {obs['target_energy_density_whkg'][0]:.0f} Wh/kg",
        f"  Minimum cycle life:     {obs['min_cycle_life'][0]:.0f} cycles to 80% capacity",
        f"  Max peak temperature:   {obs['max_peak_temp_c'][0]:.0f} °C",
        f"  Cost budget:            ${obs['max_cost_kwh'][0]:.0f}/kWh",
        f"  Charge rate:            {obs['c_rate_charge'][0]:.1f}C",
        f"  Discharge rate:         {obs['c_rate_discharge'][0]:.1f}C",
    ]
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Task description:\n{obs['task_description']}\n\n"
        f"Requirements:\n" + "\n".join(req_lines) + "\n\n"
        f"Python code:\n"
    )


# ── Model integration ──────────────────────────────────────────────────────────

def get_model_completion(model_name: str, prompt: str) -> str:
    """
    Return cell design Python code from the model.

    Replace this function (or add new elif branches) with your LLM call.
    The returned string must define a `params` dict.
    """
    if model_name == "dummy":
        # Minimal valid design — used to test the harness plumbing.
        # Deliberately naive (zero-shot-style) so the harness shows fails on hard tasks.
        return """
params = {
    "chemistry": "NMC532",
    "negative_electrode_thickness": 50e-6,
    "negative_electrode_porosity": 0.50,
    "negative_electrode_particle_radius": 10e-6,
    "positive_electrode_thickness": 50e-6,
    "positive_electrode_porosity": 0.50,
    "positive_electrode_particle_radius": 8e-6,
    "separator_thickness": 30e-6,
    "separator_porosity": 0.50,
    "ambient_temperature_celsius": 25.0,
}
"""
    if model_name == "dummy_optimised":
        # Better design — tests that the harness can record passes.
        return """
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
    raise NotImplementedError(
        f"Model '{model_name}' not implemented. "
        "Edit get_model_completion() in this file to integrate your LLM."
    )


# ── Benchmark runner ───────────────────────────────────────────────────────────

def run_benchmark(
    model_name: str,
    difficulty: str | None = None,
    dataset_path: str | None = None,
    output_path: str | None = None,
    evaluator: str = "anonymous",
    hardware: str = "local",
) -> dict:
    """
    Evaluate `model_name` on all (or filtered) CellDesign tasks.

    Returns the full results dict (also written to JSON if output_path given).
    """
    # Build env kwargs
    env_kwargs: dict = {}
    if dataset_path:
        env_kwargs["mode"] = "dataset"
        env_kwargs["dataset_path"] = dataset_path

    env = gym.make("prodata/CellDesign-v0", **env_kwargs)

    with open(TASKS_PATH) as f:
        all_tasks = json.load(f)

    if difficulty:
        tasks = [t for t in all_tasks if t["difficulty"] == difficulty]
        if not tasks:
            raise ValueError(f"No tasks found for difficulty={difficulty!r}")
    else:
        tasks = all_tasks

    print(f"Evaluating {model_name} on {len(tasks)} tasks"
          + (f" [{difficulty}]" if difficulty else "")
          + (" [dataset mode]" if dataset_path else " [live PyBaMM]"))
    print()

    task_results: dict[str, dict] = {}
    per_difficulty: dict[str, list] = defaultdict(list)
    dim_totals: dict[str, list] = defaultdict(list)
    t0 = time.time()

    for i, task in enumerate(tasks):
        obs, _ = env.reset(options={"task_id": task["task_id"]})

        prompt = build_prompt(obs)
        code = get_model_completion(model_name, prompt)

        _, reward, _, _, info = env.step(code)

        passed = info["success"]
        dim_scores = info["dimension_scores"]  # dict[str, float]

        task_results[task["task_id"]] = {
            "passed": passed,
            "reward": round(float(reward), 4),
            "dimension_scores": {k: round(float(v), 4) for k, v in dim_scores.items()},
        }
        per_difficulty[task["difficulty"]].append(passed)
        for k, v in dim_scores.items():
            dim_totals[k].append(float(v))

        status = "PASS" if passed else "FAIL"
        print(f"  [{i+1:2d}/{len(tasks)}] {task['task_id']:<40} "
              f"{reward:.3f}  {status}")

    env.close()
    elapsed = time.time() - t0

    # ── Aggregate ────────────────────────────────────────────────────────────
    all_passed = [r["passed"] for r in task_results.values()]
    overall_pass_rate = sum(all_passed) / len(all_passed) if all_passed else 0.0

    pass_rate_by_diff = {
        diff: round(sum(flags) / len(flags), 3) if flags else 0.0
        for diff, flags in per_difficulty.items()
    }

    mean_scores = {
        k: round(sum(v) / len(v), 4)
        for k, v in dim_totals.items()
    }

    result = {
        "model": model_name,
        "model_params": None,   # fill in before PR: "7B", "32B", etc.
        "rl_trained": False,    # set to True if RL-trained
        "rl_episodes": None,    # set to int if RL-trained
        "eval_date": str(date.today()),
        "evaluator": evaluator,
        "hardware": hardware,
        "overall_pass_rate": round(overall_pass_rate, 3),
        "pass_rate_by_difficulty": pass_rate_by_diff,
        "mean_scores": mean_scores,
        "n_tasks": len(tasks),
        "elapsed_seconds": round(elapsed, 1),
        "dataset_mode": dataset_path is not None,
        "task_results": task_results,
    }

    # ── Print summary ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Model:       {model_name}")
    print(f"  Tasks:       {len(tasks)}")
    print(f"  Pass rate:   {overall_pass_rate*100:.1f}%  "
          f"({sum(all_passed)}/{len(all_passed)} tasks)")
    for diff in ("easy", "medium", "hard"):
        if diff in pass_rate_by_diff:
            flags = per_difficulty[diff]
            print(f"    {diff:8}  {pass_rate_by_diff[diff]*100:.0f}%  "
                  f"({sum(flags)}/{len(flags)})")
    print(f"  Dim scores:  "
          + "  ".join(f"{k}={v:.3f}" for k, v in mean_scores.items()))
    print(f"  Elapsed:     {elapsed:.1f}s")
    print("=" * 60)

    # ── Save JSON ────────────────────────────────────────────────────────────
    if output_path is None:
        out_dir = Path("benchmarks/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fname = f"{model_name.replace('/', '_')}_{date.today()}.json"
        output_path = str(out_dir / out_fname)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"\nResults saved: {output_path}")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the Prodata battery benchmark."
    )
    parser.add_argument(
        "--model", default="dummy",
        help="Model name. Use 'dummy' or 'dummy_optimised' to test the harness."
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"], default=None,
        help="Restrict to one difficulty tier."
    )
    parser.add_argument(
        "--dataset", default=None, metavar="PATH",
        help="Path to pre-computed Parquet dataset (enables ~1ms reward lookups)."
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Output JSON path. Default: benchmarks/results/<model>_<date>.json"
    )
    parser.add_argument(
        "--evaluator", default="anonymous",
        help="GitHub handle or name to include in the JSON."
    )
    parser.add_argument(
        "--hardware", default="local",
        help="Hardware description (e.g. 'Colab T4', 'local A100')."
    )
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        difficulty=args.difficulty,
        dataset_path=args.dataset,
        output_path=args.output,
        evaluator=args.evaluator,
        hardware=args.hardware,
    )

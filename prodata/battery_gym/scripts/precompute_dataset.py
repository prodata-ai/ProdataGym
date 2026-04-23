"""
Pre-compute battery simulation dataset for fast RL training.

Uses Latin Hypercube Sampling across the 10-dimensional electrode parameter space,
runs real PyBaMM simulations in parallel, and saves results as Parquet.

During RL training, load the dataset and use mode="dataset" in CellDesignEnv
for ~1 ms/step instead of ~3 s/step in live mode.

Usage:
    # Recommended: 10k samples, 8 workers (~20-40 minutes on modern hardware)
    python -m prodata.battery_gym.scripts.precompute_dataset \\
        --samples 10000 \\
        --workers 8 \\
        --output data/battery_dataset_10k.parquet

    # Quick test run (5 minutes)
    python -m prodata.battery_gym.scripts.precompute_dataset \\
        --samples 500 \\
        --workers 4 \\
        --output data/battery_dataset_test.parquet

    # Then use in training:
    env = gym.make("prodata/CellDesign-v0", mode="dataset",
                   dataset_path="data/battery_dataset_10k.parquet")

Parameter ranges sampled (Latin Hypercube):
    negative_electrode_thickness    [20, 200] µm
    negative_electrode_porosity     [0.10, 0.65]
    negative_electrode_particle_radius  [0.5, 20] µm
    positive_electrode_thickness    [20, 180] µm
    positive_electrode_porosity     [0.10, 0.65]
    positive_electrode_particle_radius  [0.5, 15] µm
    separator_thickness             [10, 60] µm
    separator_porosity              [0.25, 0.70]
    c_rate_discharge                [0.5, 3.0]
    ambient_temperature_celsius     [-20, 45]

Chemistry: NMC532 (60%), LFP (40%)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------

PARAM_NAMES = [
    "negative_electrode_thickness",
    "negative_electrode_porosity",
    "negative_electrode_particle_radius",
    "positive_electrode_thickness",
    "positive_electrode_porosity",
    "positive_electrode_particle_radius",
    "separator_thickness",
    "separator_porosity",
    "c_rate_discharge",
    "ambient_temperature_celsius",
]

PARAM_LO = np.array([
    20e-6,   # neg thickness m
    0.10,    # neg porosity
    0.5e-6,  # neg particle radius m
    20e-6,   # pos thickness m
    0.10,    # pos porosity
    0.5e-6,  # pos particle radius m
    10e-6,   # sep thickness m
    0.25,    # sep porosity
    0.5,     # c_rate_discharge
    -20.0,   # ambient temp °C
])

PARAM_HI = np.array([
    200e-6,  # neg thickness m
    0.65,    # neg porosity
    20e-6,   # neg particle radius m
    180e-6,  # pos thickness m
    0.65,    # pos porosity
    15e-6,   # pos particle radius m
    60e-6,   # sep thickness m
    0.70,    # sep porosity
    3.0,     # c_rate_discharge
    45.0,    # ambient temp °C
])

OUTPUT_COLS = [
    "energy_density_whkg",
    "capacity_ah",
    "peak_temperature_c",
    "cycle_life_80pct",
    "estimated_cost_kwh",
    "success",
]


# ---------------------------------------------------------------------------
# Worker function (runs in a separate process)
# ---------------------------------------------------------------------------

def _simulate_one(args: tuple) -> dict[str, Any]:
    """
    Run one PyBaMM simulation. Executed in a worker process.

    Returns a dict with all parameter values + simulation outputs.
    """
    sample_id, params, chemistry, c_rate_charge = args

    # Import here — each worker process gets its own PyBaMM instance
    from prodata.battery_gym.simulators.electrochemical_sim import ElectrochemicalSimulator

    sim = ElectrochemicalSimulator(mode="live")

    task_spec = {
        "c_rate_discharge": params["c_rate_discharge"],
        "c_rate_charge":    c_rate_charge,
    }

    code = f"""
params = {{
    "chemistry":                         {chemistry!r},
    "negative_electrode_thickness":       {params['negative_electrode_thickness']!r},
    "negative_electrode_porosity":        {params['negative_electrode_porosity']!r},
    "negative_electrode_particle_radius": {params['negative_electrode_particle_radius']!r},
    "positive_electrode_thickness":       {params['positive_electrode_thickness']!r},
    "positive_electrode_porosity":        {params['positive_electrode_porosity']!r},
    "positive_electrode_particle_radius": {params['positive_electrode_particle_radius']!r},
    "separator_thickness":                {params['separator_thickness']!r},
    "separator_porosity":                 {params['separator_porosity']!r},
    "ambient_temperature_celsius":        {params['ambient_temperature_celsius']!r},
}}
"""

    result = sim.execute(code, task_spec)

    row: dict[str, Any] = {
        "sample_id":   sample_id,
        "chemistry":   chemistry,
        **params,
        "c_rate_charge": c_rate_charge,
        "success":     result.success,
    }

    if result.success:
        out = result.outputs
        row["energy_density_whkg"] = out.get("energy_density_whkg", 0.0)
        row["capacity_ah"]         = out.get("capacity_ah", 0.0)
        row["peak_temperature_c"]  = out.get("peak_temperature_c", 0.0)
        row["cycle_life_80pct"]    = float(out.get("cycle_life_80pct", 0))
        row["estimated_cost_kwh"]  = out.get("estimated_cost_kwh", 0.0)
    else:
        row["energy_density_whkg"] = 0.0
        row["capacity_ah"]         = 0.0
        row["peak_temperature_c"]  = 0.0
        row["cycle_life_80pct"]    = 0.0
        row["estimated_cost_kwh"]  = 0.0
        row["error"]               = result.error

    return row


# ---------------------------------------------------------------------------
# Main dataset generation
# ---------------------------------------------------------------------------

def precompute_dataset(
    n_samples: int = 10_000,
    output_path: str | Path = "data/battery_dataset_10k.parquet",
    n_workers: int | None = None,
    seed: int = 42,
    chemistry_split: float = 0.60,   # fraction NMC532 vs LFP
) -> None:
    """
    Generate the pre-computed battery dataset.

    Args:
        n_samples:       Total number of simulations to run
        output_path:     Where to save the Parquet file
        n_workers:       Parallel workers (default: cpu_count - 1)
        seed:            Random seed for reproducibility
        chemistry_split: Fraction of samples that use NMC532 (rest use LFP)
    """
    import pandas as pd
    from scipy.stats.qmc import LatinHypercube

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"Generating {n_samples} samples using {n_workers} workers")
    print(f"Output: {output_path}")
    print()

    # Latin Hypercube Sampling over 10 continuous parameters
    rng = np.random.default_rng(seed)
    sampler = LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit_samples = sampler.random(n=n_samples)

    # Scale from [0, 1] to physical parameter ranges
    scaled = PARAM_LO + unit_samples * (PARAM_HI - PARAM_LO)

    # Assign chemistries
    n_nmc = int(n_samples * chemistry_split)
    chemistries = ["NMC532"] * n_nmc + ["LFP"] * (n_samples - n_nmc)
    rng.shuffle(chemistries)

    # C-rate for charging: sample independently [0.5, 4.0]
    c_rates_charge = rng.uniform(0.5, 4.0, size=n_samples)

    # Build argument list for workers
    work_items = []
    for i in range(n_samples):
        params = {name: float(scaled[i, j]) for j, name in enumerate(PARAM_NAMES)}
        work_items.append((i, params, chemistries[i], float(c_rates_charge[i])))

    # Run in parallel
    t0 = time.time()
    rows = []
    failed = 0

    ctx = mp.get_context("spawn")  # spawn avoids PyBaMM state issues with fork
    with ctx.Pool(n_workers, maxtasksperchild=100) as pool:
        for i, row in enumerate(pool.imap_unordered(_simulate_one, work_items, chunksize=10)):
            rows.append(row)
            if not row.get("success", False):
                failed += 1

            if (i + 1) % 100 == 0 or (i + 1) == n_samples:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (n_samples - i - 1) / rate if rate > 0 else 0
                print(
                    f"  {i+1:>6}/{n_samples}  |  "
                    f"{rate:.1f} sims/s  |  "
                    f"ETA {remaining/60:.1f} min  |  "
                    f"failed: {failed}"
                )

    elapsed_total = time.time() - t0
    print(f"\nCompleted {n_samples} simulations in {elapsed_total/60:.1f} minutes")
    print(f"Success rate: {(n_samples - failed) / n_samples * 100:.1f}%")

    # Build DataFrame and save
    df = pd.DataFrame(rows)

    # Drop failed rows for the clean dataset
    df_clean = df[df["success"] == True].copy()
    print(f"Rows saved: {len(df_clean)} / {n_samples}")

    df_clean.to_parquet(output_path, index=False, compression="snappy")
    print(f"Saved to {output_path}  ({output_path.stat().st_size / 1024:.0f} KB)")

    # Save summary statistics
    print("\nOutput statistics:")
    for col in ["energy_density_whkg", "cycle_life_80pct", "peak_temperature_c", "estimated_cost_kwh"]:
        if col in df_clean.columns:
            print(f"  {col:30s}  min={df_clean[col].min():.1f}  "
                  f"mean={df_clean[col].mean():.1f}  max={df_clean[col].max():.1f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute battery simulation dataset for fast RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--samples",  type=int,   default=10_000,
                        help="Number of simulations to run (default: 10000)")
    parser.add_argument("--workers",  type=int,   default=None,
                        help="Parallel workers (default: cpu_count - 1)")
    parser.add_argument("--output",   type=str,   default="data/battery_dataset_10k.parquet",
                        help="Output Parquet file path")
    parser.add_argument("--seed",     type=int,   default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    precompute_dataset(
        n_samples=args.samples,
        output_path=args.output,
        n_workers=args.workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

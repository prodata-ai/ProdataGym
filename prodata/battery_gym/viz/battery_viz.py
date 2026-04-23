"""
Battery gym visualisation utilities.

All functions return matplotlib Figure objects and optionally save to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BEFORE_COLOR = "#e74c3c"   # red — zero-shot / before training
_AFTER_COLOR  = "#27ae60"   # green — after RL training
_GRID_ALPHA   = 0.25


def _get_mpl():
    """Import matplotlib lazily."""
    import matplotlib
    import matplotlib.pyplot as plt
    return matplotlib, plt


# ---------------------------------------------------------------------------
# 1. Discharge curve overlay
# ---------------------------------------------------------------------------

def plot_discharge_curves(
    before: dict,
    after: dict,
    task_description: str = "",
    save_path: str | Path | None = None,
):
    """
    Overlay discharge voltage curves for zero-shot vs RL-trained design.

    Args:
        before: dict with keys "capacity_ah", "voltage_v" from sim_result.outputs["discharge_curve"]
        after:  same format
        task_description: short task name for title
        save_path: if given, saves figure to this path

    Returns:
        matplotlib.figure.Figure
    """
    mpl, plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(8, 5))

    if before.get("capacity_ah") and before.get("voltage_v"):
        ax.plot(
            before["capacity_ah"], before["voltage_v"],
            color=_BEFORE_COLOR, linewidth=2.5, label="Zero-shot (before RL)",
            linestyle="--",
        )

    if after.get("capacity_ah") and after.get("voltage_v"):
        ax.plot(
            after["capacity_ah"], after["voltage_v"],
            color=_AFTER_COLOR, linewidth=2.5, label="After RL training",
        )

    ax.set_xlabel("Discharge Capacity (Ah)", fontsize=12)
    ax.set_ylabel("Terminal Voltage (V)", fontsize=12)
    title = "Discharge Curve"
    if task_description:
        title += f"\n{task_description[:80]}"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=_GRID_ALPHA)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 2. Capacity degradation over cycles (extrapolated)
# ---------------------------------------------------------------------------

def plot_degradation(
    before_outputs: dict,
    after_outputs: dict,
    n_cycles: int = 1000,
    save_path: str | Path | None = None,
):
    """
    Plot estimated capacity fade vs cycle number for both designs.

    Uses simple exponential fit calibrated to the simulated cycle_life_80pct values.

    Args:
        before_outputs: sim_result.outputs dict from zero-shot design
        after_outputs:  sim_result.outputs dict from RL-trained design
        n_cycles:       number of cycles to project
        save_path:      optional save path

    Returns:
        matplotlib.figure.Figure
    """
    mpl, plt = _get_mpl()

    def _fade_curve(cycle_life_80: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Exponential capacity fade: C(n) = exp(-k*n)
        k chosen so C(cycle_life_80) = 0.80
        """
        if cycle_life_80 <= 0:
            cycle_life_80 = 1
        k = -np.log(0.80) / cycle_life_80
        cycles = np.linspace(0, n, 300)
        capacity = np.exp(-k * cycles)
        return cycles, capacity

    fig, ax = plt.subplots(figsize=(8, 5))

    cycles_b, cap_b = _fade_curve(before_outputs.get("cycle_life_80pct", 200), n_cycles)
    cycles_a, cap_a = _fade_curve(after_outputs.get("cycle_life_80pct", 200), n_cycles)

    ax.plot(cycles_b, cap_b * 100, color=_BEFORE_COLOR, linewidth=2.5,
            linestyle="--", label="Zero-shot (before RL)")
    ax.plot(cycles_a, cap_a * 100, color=_AFTER_COLOR, linewidth=2.5,
            label="After RL training")

    # 80% retention line
    ax.axhline(80, color="black", linewidth=1, linestyle=":", alpha=0.5, label="80% threshold")

    # Mark cycle life crossings
    cl_b = before_outputs.get("cycle_life_80pct", 0)
    cl_a = after_outputs.get("cycle_life_80pct", 0)
    if 0 < cl_b <= n_cycles:
        ax.axvline(cl_b, color=_BEFORE_COLOR, linewidth=1, linestyle=":", alpha=0.6)
        ax.annotate(f"{cl_b} cycles", xy=(cl_b, 80), xytext=(cl_b + n_cycles * 0.02, 75),
                    color=_BEFORE_COLOR, fontsize=9)
    if 0 < cl_a <= n_cycles:
        ax.axvline(cl_a, color=_AFTER_COLOR, linewidth=1, linestyle=":", alpha=0.6)
        ax.annotate(f"{cl_a} cycles", xy=(cl_a, 80), xytext=(cl_a + n_cycles * 0.02, 83),
                    color=_AFTER_COLOR, fontsize=9)

    ax.set_xlabel("Cycle Number", fontsize=12)
    ax.set_ylabel("Capacity Retention (%)", fontsize=12)
    ax.set_title("Estimated Capacity Fade", fontsize=12)
    ax.set_ylim(50, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=_GRID_ALPHA)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 3. Dimension score radar chart
# ---------------------------------------------------------------------------

def plot_dimension_radar(
    before_scores: dict[str, float],
    after_scores: dict[str, float],
    save_path: str | Path | None = None,
):
    """
    Radar (spider) chart comparing dimension scores before and after RL.

    Args:
        before_scores: {"energy": 0.3, "cycle_life": 0.2, "thermal": 0.8, "cost": 0.9}
        after_scores:  same format
        save_path:     optional save path

    Returns:
        matplotlib.figure.Figure
    """
    mpl, plt = _get_mpl()

    dimensions = list(before_scores.keys())
    n = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    def _make_values(scores: dict) -> list[float]:
        vals = [scores.get(d, 0.0) for d in dimensions]
        return vals + vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    # Reference circles
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [r] * (n + 1), "k-", linewidth=0.4, alpha=0.3)

    # Before
    vals_b = _make_values(before_scores)
    ax.plot(angles, vals_b, color=_BEFORE_COLOR, linewidth=2, linestyle="--",
            label="Zero-shot")
    ax.fill(angles, vals_b, color=_BEFORE_COLOR, alpha=0.10)

    # After
    vals_a = _make_values(after_scores)
    ax.plot(angles, vals_a, color=_AFTER_COLOR, linewidth=2, label="After RL")
    ax.fill(angles, vals_a, color=_AFTER_COLOR, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace("_", "\n") for d in dimensions], fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title("Dimension Scores", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 4. Score distribution histogram (training curve substitute)
# ---------------------------------------------------------------------------

def plot_score_distribution(
    before_scores: list[float],
    after_scores: list[float],
    save_path: str | Path | None = None,
):
    """
    Histogram showing the distribution of overall scores before vs after training.

    Args:
        before_scores: list of overall_score floats from zero-shot evaluation
        after_scores:  list from post-training evaluation
        save_path:     optional save path

    Returns:
        matplotlib.figure.Figure
    """
    mpl, plt = _get_mpl()
    bins = np.linspace(0, 1, 21)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(before_scores, bins=bins, alpha=0.65, color=_BEFORE_COLOR,
            label=f"Zero-shot  (mean={np.mean(before_scores):.2f})", edgecolor="white")
    ax.hist(after_scores,  bins=bins, alpha=0.65, color=_AFTER_COLOR,
            label=f"After RL   (mean={np.mean(after_scores):.2f})", edgecolor="white")

    ax.axvline(0.70, color="black", linewidth=1.5, linestyle=":", label="Pass threshold (0.70)")
    ax.set_xlabel("Overall Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Score Distribution: Before vs After RL Training", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=_GRID_ALPHA)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 5. Pareto scatter: energy density vs cycle life
# ---------------------------------------------------------------------------

def plot_pareto(
    before_results: list[dict],
    after_results: list[dict],
    task_target_energy: float | None = None,
    task_target_cycles: int | None = None,
    save_path: str | Path | None = None,
):
    """
    Scatter plot of energy density vs cycle life for two sets of designs.

    Shows that RL training moves designs toward the Pareto frontier.

    Args:
        before_results: list of sim_result.outputs dicts (zero-shot)
        after_results:  list of sim_result.outputs dicts (after RL)
        task_target_energy:  if given, draws a vertical target line
        task_target_cycles:  if given, draws a horizontal target line
        save_path:     optional save path

    Returns:
        matplotlib.figure.Figure
    """
    mpl, plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(8, 6))

    def _scatter(results, color, label, marker):
        xs = [r.get("energy_density_whkg", 0) for r in results]
        ys = [r.get("cycle_life_80pct", 0) for r in results]
        ax.scatter(xs, ys, c=color, alpha=0.5, s=30, marker=marker, label=label)

    _scatter(before_results, _BEFORE_COLOR, "Zero-shot", "x")
    _scatter(after_results,  _AFTER_COLOR,  "After RL",  "o")

    if task_target_energy:
        ax.axvline(task_target_energy, color="navy", linewidth=1.2, linestyle="--",
                   alpha=0.6, label=f"Energy target ({task_target_energy} Wh/kg)")
    if task_target_cycles:
        ax.axhline(task_target_cycles, color="darkgreen", linewidth=1.2, linestyle="--",
                   alpha=0.6, label=f"Cycle target ({task_target_cycles})")

    ax.set_xlabel("Energy Density (Wh/kg)", fontsize=12)
    ax.set_ylabel("Estimated Cycle Life (cycles to 80%)", fontsize=12)
    ax.set_title("Pareto Landscape: Energy Density vs Cycle Life", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=_GRID_ALPHA)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 6. Combined before/after panel (the "wow" figure)
# ---------------------------------------------------------------------------

def plot_before_after(
    before_sim_outputs: dict,
    after_sim_outputs: dict,
    before_scores: dict[str, float],
    after_scores: dict[str, float],
    task_description: str = "",
    save_path: str | Path | None = None,
):
    """
    2×2 panel showing the full before/after comparison:
      - Top left:  Discharge curves
      - Top right: Capacity degradation
      - Bottom left:  Score radar
      - Bottom right: Metrics bar chart

    Args:
        before_sim_outputs: sim_result.outputs dict for zero-shot design
        after_sim_outputs:  sim_result.outputs dict for RL-trained design
        before_scores:      dimension_scores dict from verifier (before)
        after_scores:       dimension_scores dict from verifier (after)
        task_description:   short string for the suptitle
        save_path:          optional save path

    Returns:
        matplotlib.figure.Figure
    """
    mpl, plt = _get_mpl()
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.35)

    ax_discharge = fig.add_subplot(gs[0, 0])
    ax_degrade   = fig.add_subplot(gs[0, 1])
    ax_radar     = fig.add_subplot(gs[1, 0], polar=True)
    ax_bars      = fig.add_subplot(gs[1, 1])

    # ---- Discharge curves ----
    bc = before_sim_outputs.get("discharge_curve", {})
    ac = after_sim_outputs.get("discharge_curve", {})
    if bc.get("capacity_ah") and bc.get("voltage_v"):
        ax_discharge.plot(bc["capacity_ah"], bc["voltage_v"],
                          color=_BEFORE_COLOR, linewidth=2, linestyle="--",
                          label="Zero-shot")
    if ac.get("capacity_ah") and ac.get("voltage_v"):
        ax_discharge.plot(ac["capacity_ah"], ac["voltage_v"],
                          color=_AFTER_COLOR, linewidth=2, label="After RL")
    ax_discharge.set_xlabel("Capacity (Ah)")
    ax_discharge.set_ylabel("Voltage (V)")
    ax_discharge.set_title("Discharge Curve")
    ax_discharge.legend(fontsize=9)
    ax_discharge.grid(True, alpha=_GRID_ALPHA)

    # ---- Degradation ----
    n_plot_cycles = max(
        before_sim_outputs.get("cycle_life_80pct", 500),
        after_sim_outputs.get("cycle_life_80pct", 500),
        500,
    )

    def _fade(cl80):
        if cl80 <= 0:
            cl80 = 1
        k = -np.log(0.80) / cl80
        cycles = np.linspace(0, n_plot_cycles, 300)
        return cycles, np.exp(-k * cycles) * 100

    cyc_b, cap_b = _fade(before_sim_outputs.get("cycle_life_80pct", 200))
    cyc_a, cap_a = _fade(after_sim_outputs.get("cycle_life_80pct", 200))
    ax_degrade.plot(cyc_b, cap_b, color=_BEFORE_COLOR, linewidth=2, linestyle="--",
                    label="Zero-shot")
    ax_degrade.plot(cyc_a, cap_a, color=_AFTER_COLOR, linewidth=2, label="After RL")
    ax_degrade.axhline(80, color="k", linewidth=1, linestyle=":", alpha=0.5)
    ax_degrade.set_xlabel("Cycle Number")
    ax_degrade.set_ylabel("Capacity Retention (%)")
    ax_degrade.set_title("Estimated Capacity Fade")
    ax_degrade.set_ylim(50, 105)
    ax_degrade.legend(fontsize=9)
    ax_degrade.grid(True, alpha=_GRID_ALPHA)

    # ---- Radar ----
    dimensions = list(before_scores.keys())
    n = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    def _vals(scores):
        v = [scores.get(d, 0) for d in dimensions]
        return v + v[:1]

    for r in [0.25, 0.5, 0.75, 1.0]:
        ax_radar.plot(angles, [r] * (n + 1), "k-", lw=0.4, alpha=0.3)
    ax_radar.plot(angles, _vals(before_scores), color=_BEFORE_COLOR, lw=2,
                  linestyle="--", label="Zero-shot")
    ax_radar.fill(angles, _vals(before_scores), color=_BEFORE_COLOR, alpha=0.10)
    ax_radar.plot(angles, _vals(after_scores), color=_AFTER_COLOR, lw=2, label="After RL")
    ax_radar.fill(angles, _vals(after_scores), color=_AFTER_COLOR, alpha=0.15)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([d.replace("_", "\n") for d in dimensions], fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([])
    ax_radar.set_title("Dimension Scores", fontsize=11, pad=18)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)

    # ---- Metrics bar chart ----
    metrics = {
        "Energy\n(Wh/kg)":       (before_sim_outputs.get("energy_density_whkg", 0),
                                   after_sim_outputs.get("energy_density_whkg", 0)),
        "Cycle Life\n(÷10)":     (before_sim_outputs.get("cycle_life_80pct", 0) / 10,
                                   after_sim_outputs.get("cycle_life_80pct", 0) / 10),
        "Peak Temp\n(°C)":       (before_sim_outputs.get("peak_temperature_c", 0),
                                   after_sim_outputs.get("peak_temperature_c", 0)),
        "Cost\n($/kWh)":         (before_sim_outputs.get("estimated_cost_kwh", 0),
                                   after_sim_outputs.get("estimated_cost_kwh", 0)),
    }
    x = np.arange(len(metrics))
    w = 0.35
    labels = list(metrics.keys())
    b_vals = [v[0] for v in metrics.values()]
    a_vals = [v[1] for v in metrics.values()]
    ax_bars.bar(x - w / 2, b_vals, w, color=_BEFORE_COLOR, alpha=0.8, label="Zero-shot")
    ax_bars.bar(x + w / 2, a_vals, w, color=_AFTER_COLOR, alpha=0.8, label="After RL")
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(labels, fontsize=9)
    ax_bars.set_title("Key Metrics Comparison", fontsize=11)
    ax_bars.legend(fontsize=9)
    ax_bars.grid(True, axis="y", alpha=_GRID_ALPHA)

    title = "Battery Cell Design: Before vs After RL Training"
    if task_description:
        title += f"\n{task_description[:100]}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

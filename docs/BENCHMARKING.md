# Benchmarking & Visualization Guide

What to measure, what charts to plot, and how to interpret them.

---

## The five charts that matter

### 1. Pass rate over training episodes (learning curve)

**What it shows:** Is the model actually improving?

```
Pass rate (%)
100% ┤                                          ╭──── Post-RL plateau
 80% ┤                                    ╭────╯
 60% ┤                              ╭────╯
 40% ┤                       ╭─────╯
 20% ┤         ╭─────────────╯
  0% ┤─────────╯ Zero-shot baseline
     └────────────────────────────────────────▶ Episodes
     0        250       500       750      1000
```

**How to generate:**
```python
import matplotlib.pyplot as plt

episodes = [0, 100, 200, 300, 400, 500]
pass_rates = [14, 22, 35, 48, 60, 67]   # example

plt.figure(figsize=(8, 4))
plt.plot(episodes, pass_rates, marker='o', linewidth=2)
plt.axhline(y=pass_rates[0], linestyle='--', color='gray', label='Zero-shot baseline')
plt.xlabel("Training episodes")
plt.ylabel("Pass rate (%)")
plt.title("BracketDesign-v0 — Pass rate over RL training")
plt.legend()
plt.tight_layout()
plt.savefig("results/learning_curve.png", dpi=150)
```

---

### 2. Dimension score breakdown — radar chart

**What it shows:** Where the model is strong vs. weak. Is it gaming one dimension?

```
         Structural
              1.0
             /   \
           0.5   0.5
          /         \
Geometry ──────────── Cost
```

**How to generate:**
```python
import numpy as np
import matplotlib.pyplot as plt

labels = ["Structural", "Cost", "Geometry"]
zero_shot = [0.38, 0.61, 0.72]
post_rl   = [0.84, 0.55, 0.91]   # note: cost often drops as model learns to push structure up

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
for values, label, color in [(zero_shot, "Zero-shot", "steelblue"),
                              (post_rl, "Post-RL", "darkorange")]:
    v = values + values[:1]
    ax.plot(angles, v, color=color, linewidth=2, label=label)
    ax.fill(angles, v, color=color, alpha=0.15)

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 1)
ax.legend(loc="upper right")
plt.title("Score dimensions: zero-shot vs post-RL")
plt.tight_layout()
plt.savefig("results/radar_chart.png", dpi=150)
```

**Red flag:** If structural → 1.0 but cost → 0.0, gaming is happening.

---

### 3. Per-difficulty pass rate bar chart

**What it shows:** Does the model generalize to hard tasks, or only solve easy ones?

```
Pass rate
100% ┤ ████
 80% ┤ ████  ████
 60% ┤ ████  ████
 40% ┤ ████  ████  ████
 20% ┤ ████  ████  ████
  0% └───────────────────
      Easy  Medium  Hard
```

**How to generate:**
```python
import matplotlib.pyplot as plt

difficulties = ["Easy", "Medium", "Hard"]
zero_shot = [33, 12, 4]
post_rl   = [87, 65, 42]

x = range(len(difficulties))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([i - width/2 for i in x], zero_shot, width, label="Zero-shot", color="steelblue")
ax.bar([i + width/2 for i in x], post_rl,   width, label="Post-RL",   color="darkorange")
ax.set_xticks(x)
ax.set_xticklabels(difficulties)
ax.set_ylabel("Pass rate (%)")
ax.set_title("Pass rate by difficulty tier")
ax.legend()
plt.tight_layout()
plt.savefig("results/pass_by_difficulty.png", dpi=150)
```

---

### 4. Safety factor distribution (reward hacking detector)

**What it shows:** A model gaming the basic verifier will produce unphysically high safety factors. Legitimate designs cluster 1.5–6.0. Gamed designs show SF > 20.

```
Count
 40 ┤     ████
 30 ┤  ██  ████ ██
 20 ┤  ██  ████ ████ ██
 10 ┤  ██  ████ ████ ████ ██         █████ ██████ █
  0 └─────────────────────────────────────────────▶
    0   1   2   3   4   5  ...  10 ... 20 ... 50+
                                     SF
```

**How to generate:**
```python
import matplotlib.pyplot as plt

# Collect safety_factors from all episode step infos
safety_factors = [info["dimension_scores"]["structural"] for info in episode_infos]

plt.figure(figsize=(8, 4))
plt.hist(safety_factors, bins=40, color="steelblue", edgecolor="white")
plt.axvline(x=3.0, color='green', linestyle='--', label='Target SF=3.0')
plt.axvline(x=20.0, color='red', linestyle='--', label='Gaming threshold')
plt.xlabel("Safety factor")
plt.ylabel("Count")
plt.title("Safety factor distribution (gaming detector)")
plt.legend()
plt.tight_layout()
plt.savefig("results/sf_distribution.png", dpi=150)
```

---

### 5. Reward distribution — episode histogram

**What it shows:** How rewards are distributed across training. A healthy distribution shifts right over training. A gamed distribution shows a spike near 1.0 and a spike near -1.0 (crash vs. perfect fake).

```python
import matplotlib.pyplot as plt

# rewards_by_checkpoint: dict of {episode: [reward_list]}
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, (ep, rewards) in zip(axes, rewards_by_checkpoint.items()):
    ax.hist(rewards, bins=20, range=(-1, 1), color="steelblue")
    ax.set_title(f"Episode {ep}")
    ax.set_xlabel("Reward")
axes[0].set_ylabel("Count")
plt.suptitle("Reward distribution over training")
plt.tight_layout()
plt.savefig("results/reward_distribution.png", dpi=150)
```

---

## What numbers to log during training

Log these after every evaluation checkpoint:

```python
checkpoint_metrics = {
    "episode":              int,
    "pass_rate_overall":    float,    # fraction of 50 tasks passing
    "pass_rate_easy":       float,
    "pass_rate_medium":     float,
    "pass_rate_hard":       float,
    "mean_reward":          float,
    "mean_structural":      float,
    "mean_cost":            float,
    "mean_geometry":        float,
    "mean_safety_factor":   float,    # watch for inflation
    "p90_safety_factor":    float,    # top 10% — gaming shows up here first
    "fraction_crashed":     float,    # reward == -1.0
    "fraction_gamed":       float,    # SF > 20 and cost_score < 0.2
}
```

Save to `results/checkpoints/<model>_<date>.jsonl` — one JSON line per checkpoint.

---

## Full benchmark script

`examples/cad/02_run_benchmark.py` runs any model function against all 50 tasks,
saves results JSON, and prints a summary table:

```
==================================================
Model: Qwen2.5-Coder-7B  |  Zero-shot baseline
==================================================
Difficulty   Tasks   Passed   Pass%   Avg reward
──────────── ─────── ──────── ─────── ──────────
Easy            15       5    33.3%      0.521
Medium          20       2    10.0%      0.318
Hard            15       0     0.0%      0.201
──────────── ─────── ──────── ─────── ──────────
TOTAL           50       7    14.0%      0.342
==================================================
Top failure reason: structural (safety factor < 1.5)
```

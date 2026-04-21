# CAD Gym — User Guide

Everything you need to test `BracketDesign-v0` end-to-end.

---

## What this environment does

Your AI agent writes Python code (using CadQuery) to design a mechanical bracket.  
The environment executes that code, runs a simulation, and scores the result.  
The score is the RL reward signal.

---

## Install

```bash
# From the repo root
pip install -e ".[cad]"

# Or from GitHub
pip install "git+https://github.com/prodata-ai/ProdataGym.git#egg=prodata[cad]"
```

Dependencies: `cadquery`, `trimesh`, `gymnasium`, `pydantic`. No API key needed.

---

## Minimal working example

```python
import gymnasium as gym
import prodata.cad_gym  # must import to register environments

env = gym.make("prodata/BracketDesign-v0")
obs, info = env.reset()

print(obs["task_description"])
# "Design an L-bracket to mount to a vertical wall and support a 5 kg
#  static load at 100 mm horizontal extension..."

action = """
import cadquery as cq
result = cq.Workplane("XY").box(120, 80, 15)
"""

obs, reward, terminated, truncated, info = env.step(action)

print(f"Score : {reward:.3f}")
print(f"Passed: {info['success']}")
print(f"Dims  : {info['dimension_scores']}")
```

---

## The observation dict

Returned by `env.reset()` and `env.step()`.

| Key | Type | What it is |
|-----|------|------------|
| `task_description` | str | Full plain-English task spec |
| `load_kg` | float32 (1,) | Primary vertical load |
| `extension_mm` | float32 (1,) | Distance from mounting surface to load point |
| `max_cost_usd` | float32 (1,) | Budget constraint |
| `step` | int | Which step within this episode (0 on reset) |
| `safety_factor` | float32 (1,) | Last computed safety factor (-1 before first step) |
| `cost_usd` | float32 (1,) | Last computed total cost (-1 before first step) |

---

## The action

A plain Python string. Must:

1. `import cadquery as cq` at the top
2. Define a variable named **`result`** of type `cadquery.Workplane`

```python
action = """
import cadquery as cq

# Your bracket design here
result = (
    cq.Workplane("XY")
    .box(120, 80, 15)
)
"""
```

If the code raises an exception or doesn't define `result`, reward = -1.0 and the episode terminates.

---

## The reward

A float in `[-1.0, 1.0]`.

- `-1.0` — code failed to execute
- `0.0 – 1.0` — weighted score across three dimensions
- Typical passing threshold: **≥ 0.7**

### How the score is computed

```
reward = 0.50 × structural_score
       + 0.30 × cost_score
       + 0.20 × geometry_score
```

Weights vary slightly by task difficulty — hard tasks weight structural higher.

---

## What the verifier checks

### Structural (50% of score)

Runs beam-theory FEA on the generated geometry.

| Check | How |
|-------|-----|
| **Safety factor** | σ_yield / σ_bending. Target ≥ 3.0, scores linearly from 0 at SF=0 to 1.0 at SF=3.0 |
| **Tip deflection** | δ = F·L³ / (3·E·I). Must stay under `max_deflection_mm` from task spec |
| **Combined loads** | If task has `lateral_load_kg`, lateral bending stress is added to vertical |

Material properties (E-modulus, yield strength, density) come from the task spec — the agent doesn't choose the material, it's given.

### Cost (30% of score)

```
total_cost = mass_kg × material_cost_per_kg + machining_setup_cost
score = clamp(budget / total_cost)   # 1.0 if under budget, degrades above
```

The agent is penalized for using more material than needed.

### Geometry (20% of score)

Binary check: does the bounding box of the design fit within `max_bounding_box_mm` from the task spec?  
`1.0` if it fits, `0.0` if it exceeds in any dimension.

---

## The info dict

Returned by `env.step()`.

| Key | Type | What it is |
|-----|------|------------|
| `success` | bool | Did the design pass overall? (score ≥ 0.7 and structural ≥ 0.5) |
| `dimension_scores` | dict | `{"structural": float, "cost": float, "geometry": float}` |
| `gaming_detected` | bool | Always False in basic verifier |
| `warnings` | list[str] | Human-readable issues (e.g. "Low safety factor: 0.8") |
| `step` | int | Step count within episode |
| `error` | str | Present only if code execution failed |

---

## Task selection

```python
# Random task each episode (default)
obs, info = env.reset()

# Specific task
obs, info = env.reset(options={"task_id": "cad_bracket_001"})

# List all 50 task IDs
print(env.unwrapped.task_ids())
```

---

## What good vs bad designs look like

**Common failure modes:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| `safety_factor < 1.5` | Too thin in bending direction | Increase height (z-extent) perpendicular to load |
| `max_deflection_mm exceeded` | Insufficient second moment of area | Add a gusset or increase section depth |
| `cost_score < 0.3` | Massive block of material | Remove bulk, keep material in load path only |
| `geometry_score = 0.0` | Design exceeds bounding box | Check `max_bounding_box_mm` in task spec |

**Key insight:** beam-theory FEA uses the bounding box z-dimension as the section depth.  
A bracket that is taller (in the load direction) scores structurally better.  
A bracket that wastes material scores poorly on cost.

---

## Running on a different machine

No special setup. The simulator runs entirely locally — no network calls.

```bash
# On any machine with Python 3.10+
git clone https://github.com/prodata-ai/ProdataGym.git
cd ProdataGym
pip install -e ".[cad]"
python examples/cad/01_quickstart.py
```

Or run the Colab notebook directly in any browser — no GPU required.

---

## Verifier tiers

| Tier | What it checks | Gaming risk | How to use |
|------|---------------|-------------|------------|
| **Basic** (default, open source) | SF, deflection, cost, bounding box | High — bounding box can be inflated | `gym.make("prodata/BracketDesign-v0")` |
| **Pro** (API, coming soon) | + wall thickness, tool accessibility, mesh quality, gaming pattern DB | Low | `gym.make(..., verifier_mode="pro")` + API key |

The basic verifier is intentionally simple. It is suitable for:
- Initial baselines
- Checking if your agent can generate valid CadQuery at all
- Demo/research

For actual RL training, use the Pro verifier so your agent doesn't overfit to beam-theory exploits.

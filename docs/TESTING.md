# CAD Gym — Technical Testing Guide

How the module works end-to-end, what each party is responsible for,
how to run it on a second machine, and how to measure before/after RL improvement.

---

## The data flow in one episode step

```
┌─────────────────────────────────────────────────────────────────────┐
│  YOUR AGENT (LLM or script)                                         │
│                                                                     │
│  Reads:  obs["task_description"], obs["load_kg"],                   │
│          obs["extension_mm"], obs["max_cost_usd"]                   │
│                                                                     │
│  Writes: action = """                                               │
│              import cadquery as cq                                  │
│              result = cq.Workplane("XY").box(120, 80, 15)           │
│          """                                                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  env.step(action)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BracketDesignEnv  (prodata/cad_gym/envs/bracket_env.py)            │
│                                                                     │
│  1. Calls _run_step(action, task)                                   │
│  2. Passes action string + task requirements to MechanicalSimulator │
│  3. Passes SimulationResult to BasicBracketVerifier                 │
│  4. Builds next observation dict                                    │
│  5. Returns (obs, reward, terminated, truncated, info)              │
└──────────┬──────────────────────────────┬───────────────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐    ┌─────────────────────────────────────────┐
│  MechanicalSimulator │    │  BasicBracketVerifier                   │
│  mechanical_sim.py   │    │  verifiers/basic/bracket_verifier.py    │
│                      │    │                                         │
│  _safe_exec(code)    │    │  _score_structural()  → 0–1             │
│  → exec CadQuery     │    │    SF = yield / σ_bending               │
│  → gets `result`     │    │    δ vs max_deflection_mm               │
│                      │    │                                         │
│  _export_stl()       │    │  _score_cost()        → 0–1             │
│  → CadQuery → .stl   │    │    total_cost / budget                  │
│                      │    │                                         │
│  _analyze_geometry() │    │  _score_geometry()    → 0 or 1          │
│  → trimesh volume,   │    │    bounding box check                   │
│    bbox, mass        │    │                                         │
│                      │    │  weighted_score()                       │
│  _run_fea()          │    │    0.50×struct + 0.30×cost + 0.20×geom  │
│  → beam-theory σ, δ  │    │                                         │
│                      │    │  → VerificationResult                   │
│  _estimate_cost()    │    │       .passed                           │
│  → mat + process     │    │       .overall_score  (reward)          │
│                      │    │       .dimension_scores                 │
│  → SimulationResult  │    │       .warnings                         │
└──────────────────────┘    └─────────────────────────────────────────┘
```

---

## What your agent is responsible for

| Responsibility | Detail |
|---|---|
| Generate valid Python | Must start with `import cadquery as cq` |
| Define `result` | Must be a `cadquery.Workplane` object |
| Fit the bounding box | Check `obs["task_description"]` for max dimensions |
| Keep cost down | Avoid solid blocks of material; material + CNC setup is scored |
| Structural integrity | Section depth (z-extent in bounding box) drives safety factor |

The agent does **not** choose the material, load, or process — those come from the task spec.

---

## What the environment returns

After `env.step(action)`:

```python
obs = {
    "task_description": str,         # full task text
    "load_kg":          float32[1],  # vertical load
    "extension_mm":     float32[1],  # distance from wall to load point
    "max_cost_usd":     float32[1],  # budget
    "step":             int,         # episode step count
    "safety_factor":    float32[1],  # last computed SF (-1 before first step)
    "cost_usd":         float32[1],  # last computed cost (-1 before first step)
}

reward = float  # -1.0 on error, 0.0–1.0 composite score

info = {
    "success":          bool,
    "task_id":          str,
    "dimension_scores": {"structural": float, "cost": float, "geometry": float},
    "gaming_detected":  bool,    # always False in basic verifier
    "warnings":         list[str],
    "error":            str,     # only on failure
}
```

---

## Where the simulator is called

```
env.step(action)
  └─► BracketDesignEnv._run_step()           bracket_env.py:91
        └─► MechanicalSimulator.execute()    mechanical_sim.py:52
              ├─► _safe_exec(code)           base_simulator.py:_safe_exec
              ├─► _export_stl(cad_object)    mechanical_sim.py:93
              ├─► _analyze_geometry(stl)     mechanical_sim.py:99
              ├─► _run_fea(geometry, spec)   mechanical_sim.py:112
              └─► _estimate_cost(geometry)   mechanical_sim.py:156
```

No network calls. Runs 100% locally. Each step takes ~0.5–2 seconds depending on geometry complexity.

---

## Testing on a second machine

### Prerequisites

- Python 3.10+
- git

### Setup (5 commands)

```bash
git clone https://github.com/prodata-ai/ProdataGym.git
cd ProdataGym
pip install -e ".[cad]"

# Verify install
python -c "import prodata.cad_gym; import cadquery; import trimesh; print('OK')"
```

### Run the fast test suite (no CadQuery needed)

```bash
pytest tests/test_core/ tests/test_cad_gym/test_task_schema.py tests/test_cad_gym/test_verifier.py -v
```

Expected: ~53 tests, all green, ~3 seconds.

### Run the integration tests (requires CadQuery)

```bash
pytest tests/test_cad_gym/test_env.py -v -m integration
```

Expected: ~17 tests. These actually exec CadQuery code and run the FEA — takes ~30 seconds.

### Run one episode manually

```bash
python examples/cad/01_quickstart.py
```

Or open `examples/cad/01_quickstart.ipynb` in Jupyter / Colab.

---

## Which model to use

### For a quick zero-shot baseline

**Qwen2.5-Coder-7B-Instruct** — runs free on Colab T4, produces valid CadQuery most of the time.

```python
# Colab: pip install transformers accelerate
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-Coder-7B-Instruct",
                device_map="auto", torch_dtype="auto")

def generate(prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    out = pipe(msgs, max_new_tokens=512, do_sample=False)
    return out[0]["generated_text"][-1]["content"]
```

### For best zero-shot performance

**Qwen2.5-Coder-32B-Instruct** (needs A100 or API access). This is the model reported in the README benchmarks (14% zero-shot pass rate).

### For RL training

Start with 7B. Fine-tune with GRPO or REINFORCE using `reward` as the signal.
The 32B is reported to reach 67% after RL training — that's the number to beat on the leaderboard.

### Prompt template

```python
def build_prompt(obs: dict) -> str:
    return (
        "You are a mechanical engineer. Write Python code using CadQuery to design "
        "a bracket that satisfies the requirements below.\n\n"
        f"Task: {obs['task_description']}\n\n"
        "Rules:\n"
        "- Import cadquery as cq at the top\n"
        "- Define a variable named `result` of type cadquery.Workplane\n"
        "- No other output needed\n\n"
        "CadQuery code:"
    )
```

---

## Before / after RL — how to measure improvement

### Step 1 — Establish zero-shot baseline

Run every model against all 50 tasks. Record pass rate per difficulty tier.

```python
import json
from pathlib import Path
import gymnasium as gym
import prodata.cad_gym

env = gym.make("prodata/BracketDesign-v0")
task_ids = env.unwrapped.task_ids()  # all 50

results = {}
for task_id in task_ids:
    env.reset(options={"task_id": task_id})
    code = your_model_generate(build_prompt(obs))  # zero-shot
    _, reward, _, _, info = env.step(code)
    results[task_id] = {
        "passed": info["success"],
        "reward": reward,
        "dimension_scores": info["dimension_scores"],
    }

passed = sum(1 for r in results.values() if r["passed"])
print(f"Zero-shot pass rate: {passed}/{len(task_ids)} = {passed/len(task_ids)*100:.0f}%")
```

Save `results` to `results/zero_shot_<model>_<date>.json`.

### Step 2 — RL training loop

```python
N_EPISODES = 1000
for episode in range(N_EPISODES):
    obs, info = env.reset()
    code = your_model_generate(build_prompt(obs))
    _, reward, _, _, step_info = env.step(code)

    # Feed reward back to your RL algorithm
    rl_algo.update(prompt=build_prompt(obs), action=code, reward=reward)

    if episode % 100 == 0:
        # Checkpoint eval on all 50 tasks
        eval_results = run_eval(env, your_model, task_ids)
        log_checkpoint(episode, eval_results)
```

### Step 3 — Post-RL eval

Repeat Step 1 with the fine-tuned model weights. Compare the two JSON files.

### What to report

| Metric | Zero-shot | Post-RL |
|--------|-----------|---------|
| Overall pass rate | X% | Y% |
| Easy pass rate | X% | Y% |
| Medium pass rate | X% | Y% |
| Hard pass rate | X% | Y% |
| Mean structural score | 0.XX | 0.YY |
| Mean cost score | 0.XX | 0.YY |
| Mean geometry score | 0.XX | 0.YY |

### Warning: reward hacking on the basic verifier

The basic verifier scores bounding box depth (z) as the structural section depth.
A model trained long enough will learn to make very tall, thin brackets that
score 1.0 on structural but fail manufacturing feasibility. Signs of gaming:

- Structural score → 1.0, cost score → 0.0
- Safety factor climbs to 50+ (physically unrealistic)
- Geometry score drops (design keeps hitting the bounding box ceiling)

Switch to the Pro verifier before long RL runs to avoid this.

---

## Full evaluation script

See `examples/cad/02_run_benchmark.py` for a command-line harness that:
- Runs any model against all 50 tasks
- Saves results JSON
- Prints a summary table grouped by difficulty and material

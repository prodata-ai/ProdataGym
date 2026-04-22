# Integration Guide

How to plug `prodata` environments into your own training setup.

---

## Install

```bash
pip install "git+https://github.com/prodata-ai/ProdataGym.git[cad]"
```

---

## Environment contract

Every `prodata` env is a standard Gymnasium env. The only non-standard part is that **the action is a Python string** (CadQuery code), not a tensor.

```python
import gymnasium as gym
import prodata.cad_gym

env = gym.make("prodata/BracketDesign-v0")

# --- reset ---
obs, info = env.reset()
# obs keys:
#   task_description  str    — natural language spec
#   load_kg           [1]    — required load in kg
#   extension_mm      [1]    — horizontal extension in mm
#   max_cost_usd      [1]    — budget
#   safety_factor     [1]    — -1.0 until first design submitted
#   cost_usd          [1]    — -1.0 until first design submitted
#   step              int    — current step (0 on reset)
#
# info keys:
#   task_id           str    — e.g. "cad_bracket_001"

# --- step ---
code = "import cadquery as cq\nresult = cq.Workplane('XY').box(100, 60, 10)"
obs, reward, terminated, truncated, info = env.step(code)
# reward: float in [0, 1]  (0.0 = sim failure, 1.0 = perfect pass)
# info keys:
#   success           bool
#   dimension_scores  dict   — {"structural": 0.0-1.0, "cost": 0.0-1.0, "geometry": 0.0-1.0}
#   warnings          list[str]
#   mesh_file         str | None  — path to STL if sim succeeded
#   task_id           str

# --- task control ---
env.reset(options={"task_id": "cad_bracket_003"})   # specific task
task_ids = env.unwrapped.task_ids()                  # list of all 50 task IDs
```

**Reward meaning:**

| Value | Meaning |
|---|---|
| `0.0` | Simulation failed — invalid CadQuery, crashed, or geometry error |
| `0.0–0.5` | Ran but fails requirements (too weak, over budget, wrong shape) |
| `0.5–0.85` | Partial pass — meets some criteria |
| `>0.85` | Pass — meets structural, cost, and geometry requirements |
| `-1.0` | Outer exception in your training loop (treat same as 0.0) |

---

## Pattern 1 — Minimal loop (10 lines)

The simplest possible integration. Bring your own generate function.

```python
import gymnasium as gym
import prodata.cad_gym

env = gym.make("prodata/BracketDesign-v0")

for episode in range(100):
    obs, info = env.reset()
    code = my_generate(obs["task_description"])   # your model here
    obs, reward, terminated, truncated, info = env.step(code)
    print(f"ep {episode}  task={info['task_id']}  reward={reward:.3f}  pass={info['success']}")
```

---

## Pattern 2 — REINFORCE (raw PyTorch)

Self-contained training script. No extra frameworks needed.

```python
import gymnasium as gym
import prodata.cad_gym
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

# --- setup ---
env    = gym.make("prodata/BracketDesign-v0")
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL  = "Qwen/Qwen2.5-Coder-7B-Instruct"   # or TinyLlama for quick tests
tok    = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model  = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)
opt = AdamW(model.parameters(), lr=2e-5)


def build_prompt(obs: dict) -> str:
    return (
        "<|system|>\nYou are a mechanical engineer. Write CadQuery Python code "
        "to design a bracket. Output ONLY code. No explanation.\n"
        "<|user|>\n"
        f"{obs['task_description']}\n\n"
        f"Load: {obs['load_kg'][0]:.0f} kg | "
        f"Extension: {obs['extension_mm'][0]:.0f} mm | "
        f"Budget: ${obs['max_cost_usd'][0]:.0f}\n\n"
        "Rules:\n- Start with: import cadquery as cq\n"
        "- Define a variable named `result` of type cadquery.Workplane\n"
        "<|assistant|>\n```python\nimport cadquery as cq\nresult = "
    )


# --- training loop ---
model.train()
for episode in range(500):
    obs, info = env.reset()
    prompt = build_prompt(obs)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
        )

    new_tokens = gen[0][inputs["input_ids"].shape[-1]:]
    raw  = tok.decode(new_tokens, skip_special_tokens=True)
    raw  = raw.split("```")[0] if "```" in raw else raw
    code = f"import cadquery as cq\nresult = {raw.strip()}"

    _, reward, _, _, info = env.step(code)

    # REINFORCE with baseline=0.0
    # sim failures (reward=0.0) get no gradient — only successes drive updates
    if reward > 0.01:
        opt.zero_grad()
        with torch.enable_grad():
            full  = gen[0].unsqueeze(0)
            labels = full.clone()
            labels[0, :inputs["input_ids"].shape[-1]] = -100
            loss = model(input_ids=full, labels=labels).loss * reward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    if episode % 50 == 0:
        print(f"ep {episode}  reward={reward:.3f}  pass={info['success']}")
```

---

## Pattern 3 — TRL GRPO (recommended for serious runs)

GRPO is the modern RLVR approach — same idea as REINFORCE but with group-relative reward normalization, which is more stable. Requires `trl>=0.9`.

```bash
pip install "trl>=0.9" datasets
```

```python
import numpy as np
import gymnasium as gym
import prodata.cad_gym
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- build prompt dataset from tasks ---
_env = gym.make("prodata/BracketDesign-v0")

def _task_to_prompt(obs: dict) -> str:
    return (
        "<|system|>\nYou are a mechanical engineer. Write CadQuery Python code "
        "to design a bracket. Output ONLY code.\n"
        "<|user|>\n"
        f"{obs['task_description']}\n\n"
        f"Load: {obs['load_kg'][0]:.0f} kg | "
        f"Extension: {obs['extension_mm'][0]:.0f} mm | "
        f"Budget: ${obs['max_cost_usd'][0]:.0f}\n"
        "<|assistant|>\n```python\nimport cadquery as cq\nresult = "
    )

rows = []
for tid in _env.unwrapped.task_ids():
    obs, _ = _env.reset(options={"task_id": tid})
    rows.append({"prompt": _task_to_prompt(obs), "task_id": tid})

dataset = Dataset.from_list(rows)

# --- reward function: runs completion through the env ---
def bracket_reward(completions: list[str], task_id: list[str], **kwargs) -> list[float]:
    """Called by GRPOTrainer with a batch of completions."""
    rewards = []
    env = gym.make("prodata/BracketDesign-v0")
    for code_fragment, tid in zip(completions, task_id):
        code_fragment = code_fragment.split("```")[0] if "```" in code_fragment else code_fragment
        code = f"import cadquery as cq\nresult = {code_fragment.strip()}"
        env.reset(options={"task_id": tid})
        try:
            _, reward, _, _, _ = env.step(code)
        except Exception:
            reward = 0.0
        rewards.append(max(0.0, reward))   # clip negatives from outer crashes
    return rewards

# --- trainer ---
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

config = GRPOConfig(
    output_dir="./bracket-grpo",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_generations=4,          # completions per prompt per step
    max_new_tokens=300,
    temperature=0.8,
    logging_steps=10,
    save_steps=100,
    bf16=True,
)

model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
tok   = AutoTokenizer.from_pretrained(MODEL)

trainer = GRPOTrainer(
    model=model,
    processing_class=tok,
    reward_funcs=bracket_reward,
    args=config,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model("./bracket-grpo-final")
```

---

## Pattern 4 — TRL PPO (classic RL)

Use this if you want a value network alongside the policy, or if you're comparing against prior PPO baselines.

```bash
pip install "trl>=0.9"
```

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import gymnasium as gym
import prodata.cad_gym
import torch

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

tok   = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL)
ref   = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL)   # frozen reference

config = PPOConfig(
    learning_rate=1e-5,
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=1,
)
trainer = PPOTrainer(config=config, model=model, ref_model=ref, processing_class=tok)

env = gym.make("prodata/BracketDesign-v0")

for batch_idx in range(200):
    obs, info = env.reset()
    query = build_prompt(obs)                        # reuse build_prompt from Pattern 2
    query_ids = tok(query, return_tensors="pt")["input_ids"]

    response_ids = trainer.generate(
        query_ids,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tok.pad_token_id,
    )
    new_ids = response_ids[0][query_ids.shape[-1]:]
    code_fragment = tok.decode(new_ids, skip_special_tokens=True)
    code_fragment = code_fragment.split("```")[0] if "```" in code_fragment else code_fragment
    code = f"import cadquery as cq\nresult = {code_fragment.strip()}"

    _, reward, _, _, info = env.step(code)
    reward_tensor = torch.tensor([max(0.0, reward)])

    trainer.step([query_ids[0]], [new_ids], [reward_tensor])

    if batch_idx % 20 == 0:
        print(f"batch {batch_idx}  reward={reward:.3f}  pass={info['success']}")
```

---

## Scaling up

**Run on your own GPU cluster:**
```bash
# single GPU
python train_bracket_reinforce.py

# multi-GPU with accelerate (works with Pattern 2 and 3)
accelerate launch --num_processes 4 train_bracket_grpo.py
```

**Evaluate across all 50 tasks after training:**
```python
import numpy as np

env.eval()   # if your model has dropout
results = {}
for tid in env.unwrapped.task_ids():
    obs, _ = env.reset(options={"task_id": tid})
    code   = generate_code(obs, temperature=0.1)   # greedy-ish
    _, reward, _, _, info = env.step(code)
    results[tid] = {"passed": info["success"], "reward": reward}

passed = sum(r["passed"] for r in results.values())
print(f"{passed}/50 = {passed/50:.1%}  avg reward {np.mean([r['reward'] for r in results.values()]):.3f}")
```

**Submit to leaderboard:** See `benchmarks/cad_leaderboard.md`

---

## Which pattern should I use?

| Situation | Pattern |
|---|---|
| First run, just want to verify it works | Pattern 1 — minimal loop |
| You own your training loop, no extra deps | Pattern 2 — REINFORCE |
| Serious training run, want stable gradients | Pattern 3 — TRL GRPO ← recommended |
| Comparing against PPO baselines from literature | Pattern 4 — TRL PPO |

**Model recommendations:**

| Hardware | Model |
|---|---|
| Colab T4 / laptop | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Colab A100 / 1× A100 | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| 4× A100 or better | `Qwen/Qwen2.5-Coder-32B-Instruct` |

---

## Pro verifier

Switch from the included beam-theory verifier to the full FEA-backed verifier:

```python
env = gym.make("prodata/BracketDesign-v0", verifier_mode="pro")
```

The Pro verifier catches reward hacking (e.g. hollow shells that pass bounding-box checks but have no structural integrity). Contact [prodata.ai](https://prodata.ai) for API access.

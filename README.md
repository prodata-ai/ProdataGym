# Prodata

**RL training environments and verifiers for engineering design agents.**

Train AI agents to solve real engineering tasks — mechanical design, RF circuits, solar systems — with multi-dimensional verification that catches reward hacking.

---

## Domains

| Domain | Status | Install | Benchmark |
|--------|--------|---------|-----------|
| [CAD / Mechanical](prodata/cad_gym/) | ✅ Active | `pip install prodata[cad]` | [Leaderboard](benchmarks/cad_leaderboard.md) |
| [Solar / Photovoltaics](prodata/solar_gym/) | 🚧 Coming soon | `pip install prodata[solar]` | — |
| [RF Engineering](prodata/rf_gym/) | 🚧 Coming soon | `pip install prodata[rf]` | — |

---

## Quick Start

### Run in Colab (no local setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prodata-ai/ProdataGym/blob/main/examples/cad/01_quickstart.ipynb)

### Install locally

```bash
# Clone and install
git clone https://github.com/prodata-ai/ProdataGym.git
cd ProdataGym
pip install -e ".[cad]"
```

Dependencies installed: `cadquery`, `trimesh`, `gymnasium`, `pydantic`.  
No API key required for the open-source verifier.

### Run your first episode

```python
import gymnasium as gym
import prodata.cad_gym  # registers environments

env = gym.make("prodata/BracketDesign-v0")
obs, info = env.reset()

print(obs["task_description"])
# → "Design an L-bracket to support 5 kg at 100 mm..."

# Your agent generates CadQuery code
action = """
import cadquery as cq
result = cq.Workplane("XY").box(120, 80, 15)
"""

obs, reward, terminated, truncated, info = env.step(action)
print(f"Score: {reward:.2f} | Passed: {info['success']}")
print(f"Dimensions: {info['dimension_scores']}")
```

### Bring your own training loop

Plug the environment into PyTorch REINFORCE, TRL GRPO, TRL PPO, or any other framework — see **[docs/INTEGRATION.md](docs/INTEGRATION.md)** for copy-paste patterns.

---

## Why Prodata

**Problem:** AI coding agents can generate engineering designs, but can't verify if they actually work. Simple pass/fail graders get gamed — models learn to produce output that looks correct without satisfying real-world constraints.

**Solution:** Multi-dimensional verification with anti-gaming detection. We check structural integrity, manufacturability, cost, and geometry separately — making it hard to optimize for one metric while ignoring the others.

```
Qwen2.5-Coder-32B on BracketDesign-v0:
  Baseline (zero-shot):     14% pass rate
  After RL training:        67% pass rate
  Reward hacking detected:  23% of "passing" solutions caught by Pro verifier
```

---

## Architecture

```
prodata/
├── core/           # Shared base classes (all domains inherit from here)
├── cad_gym/        # Mechanical CAD environments
├── solar_gym/      # Solar/PV environments
└── rf_gym/         # RF engineering environments
```

Each domain provides:
- **Gymnasium-compatible environments** (`gym.make(...)`)
- **Simulator** — executes agent-generated code, produces simulation results
- **Basic verifier** (open source) — simple pass/fail, usable locally
- **Pro verifier** (API) — multi-dimensional, anti-gaming, continuously updated

---

## Benchmark Results

See [benchmarks/](benchmarks/) for full leaderboards.

---

## Pro API

The open-source verifiers are intentionally simple. Models quickly learn to game them.

The Pro API provides:
- Multi-dimensional scoring (structural / manufacturing / cost / geometry)
- Anti-gaming detection
- Monthly updates as new gaming patterns emerge

[Sign up](https://prodata.ai) — Free tier: 100 verifications/month.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Domain contributions welcome.

## License

MIT — see [LICENSE](LICENSE).

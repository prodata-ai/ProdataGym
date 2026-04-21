# Prodata — Architecture & Internal Dev Guide

This doc is for you (the developer). It explains every folder, every file dependency,
and how to extend the codebase safely.

---

## Repo layout

```
prodata-ai/
│
├── prodata/                    Python package (pip-installable)
│   ├── __init__.py
│   ├── core/                   Shared base classes — ALL domains depend on this
│   └── cad_gym/                CAD domain (active)
│   └── solar_gym/              Solar domain (stub)
│   └── rf_gym/                 RF domain (stub)
│
├── tasks/                      Task JSON datasets — NOT inside Python package
│   ├── cad/                    (currently tasks live inside prodata/cad_gym/tasks/)
│   ├── solar/
│   └── rf/
│
├── benchmarks/                 Leaderboard markdown files
├── examples/                   Colab notebooks + quickstart scripts
├── tests/                      Pytest test suite
├── docs/                       Extended docs (testing guide, FEA upgrade, etc.)
│
├── pyproject.toml              Single build config, optional deps per domain
├── .gitignore                  Excludes pro verifier code + pro task datasets
└── ARCHITECTURE.md             This file
```

---

## prodata/core/ — shared infrastructure

Every domain inherits from these. Do not add domain-specific logic here.

| File | Purpose | Who uses it |
|------|---------|-------------|
| `base_env.py` | `ProdataEnv(gym.Env)` — episode loop, reset/step skeleton | All domain envs |
| `base_verifier.py` | `BaseVerifier` ABC + `VerificationResult` dataclass | All verifiers |
| `base_simulator.py` | `BaseSimulator` ABC + `SimulationResult` dataclass + `_safe_exec` | All simulators |
| `task_schema.py` | `TaskSpec`, `TaskRequirements`, `GradingCriteria` Pydantic models | All task JSON loading |
| `utils/scoring.py` | `clamp`, `weighted_score`, `threshold_score`, `linear_score` | All verifiers |

**Dependency direction:** `core` ← `cad_gym`, `solar_gym`, `rf_gym`  
Core never imports from a domain package.

---

## prodata/cad_gym/ — mechanical CAD domain

```
cad_gym/
├── __init__.py           Registers gym environments (gym.make calls go here)
├── task_schema.py        BracketRequirements, BracketTaskSpec (typed fields)
├── GUIDE.md              User-facing guide
│
├── envs/
│   └── bracket_env.py    BracketDesignEnv — extends ProdataEnv
│
├── simulators/
│   └── mechanical_sim.py MechanicalSimulator — executes CadQuery, runs FEA
│
├── verifiers/
│   ├── basic/
│   │   └── bracket_verifier.py   BasicBracketVerifier (open source, committed)
│   └── pro/
│       ├── __init__.py           Stub only (committed)
│       ├── README.md             Explains pro tier (committed)
│       └── bracket_verifier_pro.py  NOT committed (.gitignored)
│
└── tasks/
    ├── brackets_basic.json       50 tasks (open source, committed)
    └── brackets_pro.json         5000+ tasks (NOT committed, .gitignored)
```

### Call flow for one episode step

```
env.step(action_code)
    │
    ├─► BracketDesignEnv._run_step(action, task)
    │       │
    │       ├─► MechanicalSimulator.execute(code, task_spec_dict)
    │       │       │
    │       │       ├─► _safe_exec(code)          # exec CadQuery code
    │       │       ├─► _export_stl(cad_object)   # CadQuery → STL file
    │       │       ├─► _analyze_geometry(stl)    # trimesh → volume, bbox, mass
    │       │       ├─► _run_fea(geometry, spec)  # beam theory → stress, SF, deflection
    │       │       └─► _estimate_cost(geometry)  # material + machining cost
    │       │           → returns SimulationResult
    │       │
    │       └─► BasicBracketVerifier.verify(sim_result, task)
    │               │
    │               ├─► _score_structural()       # SF + deflection → 0-1
    │               ├─► _score_cost()             # total_cost vs budget → 0-1
    │               ├─► _score_geometry()         # bbox check → 0 or 1
    │               └─► weighted_score()          # composite reward
    │                   → returns VerificationResult
    │
    └─► returns (obs, reward, terminated, truncated, info)
```

---

## prodata/cad_gym/simulators/mechanical_sim.py — key internals

### Material properties table

```python
MATERIAL_PROPS = {
    "aluminum_6061_t6": {"yield_mpa": 276, "E_mpa": 69_000, "density": 2.70e-6, "cost_kg": 3.50},
    "aluminum_7075_t6": {...},
    "steel_mild":       {...},
    "steel_316":        {...},
    "steel_4140":       {...},
    "titanium_6al4v":   {...},
    "pla":              {...},
    ...
}
```

Material is set by the task spec. The agent doesn't choose it.

### FEA model (beam theory)

Treats the design as a cantilever beam loaded at the tip.

```
σ_v   = (F_vertical × extension) × (h/2) / (b × h³ / 12)
σ_lat = (F_lateral  × extension) × (b/2) / (h × b³ / 12)
σ     = σ_v + σ_lat  (conservative sum)
SF    = yield_mpa / σ
δ     = F × L³ / (3 × E × I)
```

Where `b`, `h` are taken from the bounding box of the generated geometry.

**Limitation:** this uses the bounding box, not the actual cross-section. A hollow box
and a solid box with the same bounding box get the same FEA score.
This is intentional for v1 — it's fast and deterministic. See `docs/FEA_UPGRADE.md` for v2.

---

## Adding a new domain (e.g. rf_gym)

1. Create `prodata/rf_gym/task_schema.py` — subclass `TaskRequirements` with explicit typed fields
2. Create `prodata/rf_gym/simulators/rf_sim.py` — subclass `BaseSimulator`, use PySpice/scikit-rf
3. Create `prodata/rf_gym/verifiers/basic/filter_verifier.py` — subclass `BaseVerifier`
4. Create `prodata/rf_gym/envs/filter_env.py` — subclass `ProdataEnv`, pass `task_class=FilterTaskSpec`
5. Register in `prodata/rf_gym/__init__.py` with `gymnasium.register(...)`
6. Add optional deps in `pyproject.toml` under `[project.optional-dependencies]`
7. Add tasks JSON in `prodata/rf_gym/tasks/`
8. Add tests in `tests/test_rf_gym/`

The only change to core is adding `"rf": ["scikit-rf"]` to `pyproject.toml`.

---

## What is and isn't committed

### Committed (public)
- All `prodata/core/` code
- `prodata/cad_gym/` except pro verifier files
- `prodata/cad_gym/tasks/brackets_basic.json` (50 tasks)
- All tests
- All examples and docs

### NOT committed (private, .gitignored)
- `prodata/*/verifiers/pro/*.py` (except `__init__.py` and `README.md`)
- `tasks/*/pro/` and `tasks/*/*_pro.json`
- `.env` files (API keys)
- Any `*.stl` files generated during simulation

---

## Running tests

```bash
# Fast tests — no CadQuery needed
pytest tests/test_core/ tests/test_cad_gym/test_task_schema.py tests/test_cad_gym/test_verifier.py -v

# Integration tests — requires cadquery + trimesh
pytest tests/test_cad_gym/test_env.py -v -m integration

# All tests
pytest tests/ -v
```

---

## Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `PRODATA_API_KEY` | Pro verifier | Authentication for paid API |
| `PRODATA_LOG_DIR` | Episode logger | Where to write training run logs |
| `PRODATA_CUSTOMER_ID` | Episode logger | Tag logs by customer for pro verifier training |

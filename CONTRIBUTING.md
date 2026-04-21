# Contributing to Prodata

## Adding a New Domain

Each domain lives in `prodata/<domain>_gym/`. To add a new domain (e.g. `pcb_gym`):

1. **Create the package:**
   ```
   prodata/pcb_gym/
   ├── __init__.py          # register gym environments here
   ├── envs/
   │   └── <name>_env.py   # extend ProdataEnv
   ├── simulators/
   │   └── pcb_sim.py      # extend BaseSimulator
   ├── verifiers/
   │   ├── basic/
   │   │   └── <name>_verifier.py   # extend BaseVerifier
   │   └── pro/
   │       └── README.md
   └── tasks/
       └── <tasks>_basic.json
   ```

2. **Extend base classes** from `prodata.core`:
   - `ProdataEnv` — implement `_build_observation()` and `_run_step()`
   - `BaseSimulator` — implement `execute()`
   - `BaseVerifier` — implement `verify()`

3. **Add tasks** following the `TaskSpec` schema in `prodata/core/task_schema.py`

4. **Register optional deps** in `pyproject.toml` under `[project.optional-dependencies]`

5. **Add tasks dataset** under `tasks/<domain>/`

6. **Add benchmark** under `benchmarks/<domain>_leaderboard.md`

7. **Add examples** under `examples/<domain>/`

## Task Quality Guidelines

Good tasks have:
- Clear, unambiguous spec (no two interpretations)
- Verifiable outcome (pass/fail is deterministic)
- Multiple valid solutions (not just one "right answer")
- Difficulty calibrated: easy = textbook formula, hard = requires domain knowledge

## Running Tests

```bash
pip install prodata[cad,dev]
pytest tests/
```

## Code Style

```bash
ruff check .
black .
```

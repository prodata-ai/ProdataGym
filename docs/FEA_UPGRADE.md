# FEA Upgrade Path — Beam Theory → Full FEA

This doc covers what changes when you move from v1 (beam-theory) to v2 (full FEA),
which files to touch, whether it is plug-and-play, and what the tradeoffs are.

---

## Why upgrade

The v1 simulator uses the bounding box as the structural cross-section.
A hollow box and a solid box with the same outer dimensions get the same FEA score.
This is intentional for v1 — it's fast and deterministic — but it creates a gaming surface:
an agent can learn to pad the bounding box in the z-direction without adding real material.

Full FEA evaluates the actual mesh, so hollow or thin designs are properly penalized.

---

## The two versions

| | v1 (current) | v2 (upgrade target) |
|---|---|---|
| Structural model | Cantilever beam, bounding-box cross-section | FEM on actual mesh |
| Time per step | ~50 ms | ~2–10 s (CalculiX) / ~30–60 s (FEniCS) |
| Gaming surface | Bounding box inflation | Much harder to game |
| Dependencies | trimesh only | + CalculiX or FEniCS |
| Colab-compatible | Yes | CalculiX yes, FEniCS marginal |
| Deterministic | Yes | Yes |
| Open source | Yes | Yes |

---

## What to use for v2

### Option A — CalculiX (recommended)

CalculiX is an open-source FEA solver (ABAQUS-compatible input format).
It runs as a subprocess, is fast enough for RL (2–5 s per solve), and has prebuilt binaries.

```bash
# Ubuntu / Colab
apt-get install -y calculix-ccx

# macOS
brew install calculix
```

Python interface: write the `.inp` mesh file, shell out, parse `.frd` results.

### Option B — FEniCS / dolfinx

FEniCS is a Python-native FEM library. More flexible, slower, harder to install on Colab.
Better if you want to script the mesh generation and solve entirely in Python.

```bash
pip install fenics-dolfinx  # or use the official Docker image
```

### Option C — PyNite

Pure-Python 3D structural analysis library. No external dependencies.
Less accurate than CalculiX/FEniCS but faster and easier to install.
Suitable if you want v1.5 accuracy without the solver installation burden.

```bash
pip install PyNiteFEA
```

---

## Exactly which code changes are needed

All changes are confined to `prodata/cad_gym/simulators/mechanical_sim.py`.
The rest of the codebase is untouched — the upgrade is plug-and-play at the simulator boundary.

### Step 1 — Replace `_run_fea`

Current signature (keep this, change implementation only):

```python
def _run_fea(self, geometry: dict, task_spec: dict, mat: dict) -> dict:
    # Returns: {"max_stress_mpa": float, "safety_factor": float, "max_deflection_mm": float}
```

**CalculiX implementation sketch:**

```python
def _run_fea(self, geometry: dict, task_spec: dict, mat: dict) -> dict:
    import subprocess, tempfile, os
    from prodata.cad_gym.simulators._calculix_writer import write_inp, parse_frd

    stl_path = geometry["_stl_path"]   # add stl_path to geometry dict in _analyze_geometry
    load_kg  = task_spec.get("load_kg", 10)
    E_mpa    = mat["E_mpa"]
    nu       = mat.get("poisson", 0.3)
    yield_mpa = task_spec.get("yield_strength_mpa") or mat["yield_mpa"]

    with tempfile.TemporaryDirectory() as tmpdir:
        inp_path = os.path.join(tmpdir, "bracket.inp")
        write_inp(stl_path, inp_path, load_kg=load_kg, E_mpa=E_mpa, nu=nu)

        result = subprocess.run(
            ["ccx", "-i", "bracket"],
            cwd=tmpdir, capture_output=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"CalculiX failed: {result.stderr.decode()[:500]}")

        frd_path = os.path.join(tmpdir, "bracket.frd")
        fea_results = parse_frd(frd_path)   # {"max_von_mises_mpa": float, "max_disp_mm": float}

    max_stress = fea_results["max_von_mises_mpa"]
    safety_factor = yield_mpa / max_stress if max_stress > 0 else 0.0
    deflection_mm = fea_results["max_disp_mm"]

    return {
        "max_stress_mpa":    float(max_stress),
        "safety_factor":     float(safety_factor),
        "max_deflection_mm": float(deflection_mm),
    }
```

### Step 2 — Pass `stl_path` through `_analyze_geometry`

Add `_stl_path` to the geometry dict so `_run_fea` can access the mesh file:

```python
def _analyze_geometry(self, stl_path: str, mat: dict) -> dict:
    import trimesh
    mesh = trimesh.load(stl_path)
    bbox = mesh.bounds[1] - mesh.bounds[0]
    mass_kg = mesh.volume * mat["density"]
    return {
        "_stl_path":        stl_path,          # ← add this line
        "valid":            mesh.is_watertight,
        "volume_mm3":       float(mesh.volume),
        "mass_kg":          float(mass_kg),
        "bounding_box_mm":  bbox.tolist(),
        "surface_area_mm2": float(mesh.area),
    }
```

### Step 3 — Create the CalculiX writer helper

Add `prodata/cad_gym/simulators/_calculix_writer.py`:

```python
"""
Converts an STL mesh to a CalculiX .inp file and parses results.
Uses gmsh for meshing (tetrahedral elements).
"""

def write_inp(stl_path: str, inp_path: str, load_kg: float, E_mpa: float, nu: float):
    """Convert STL → CalculiX input file with boundary conditions."""
    import gmsh
    # 1. Load STL surface mesh
    # 2. Generate tet mesh with gmsh
    # 3. Write .inp with:
    #    - *NODE, *ELEMENT sections from gmsh output
    #    - *MATERIAL with E and nu
    #    - *BOUNDARY on base nodes (fixed)
    #    - *CLOAD on tip nodes (distributed load = load_kg * 9.81 / n_tip_nodes)
    #    - *STEP / *STATIC
    #    - *NODE PRINT, *EL PRINT for stress and displacement output
    raise NotImplementedError("See gmsh + CalculiX integration guide")


def parse_frd(frd_path: str) -> dict:
    """Parse CalculiX .frd results file for max von Mises stress and displacement."""
    # CalculiX .frd is a text format with NODE sections
    # Parse max displacement magnitude and max equivalent stress
    raise NotImplementedError("Parse CalculiX .frd format")
```

### Step 4 — Add to optional dependencies

In `pyproject.toml`, add a `cad-fea` extras:

```toml
[project.optional-dependencies]
cad     = ["cadquery", "trimesh", "gymnasium", "pydantic"]
cad-fea = ["cadquery", "trimesh", "gymnasium", "pydantic", "gmsh"]
# CalculiX itself is installed as a system binary, not a pip package
```

### Step 5 — Version flag in `MechanicalSimulator`

```python
class MechanicalSimulator(BaseSimulator):
    FEA_BACKEND: str = os.getenv("PRODATA_FEA_BACKEND", "beam_theory")
    # Options: "beam_theory" (v1), "calculix" (v2), "pynitefea" (v1.5)

    def _run_fea(self, geometry, task_spec, mat):
        if self.FEA_BACKEND == "calculix":
            return self._run_fea_calculix(geometry, task_spec, mat)
        elif self.FEA_BACKEND == "pynitefea":
            return self._run_fea_pynitefea(geometry, task_spec, mat)
        else:
            return self._run_fea_beam_theory(geometry, task_spec, mat)
```

Set `PRODATA_FEA_BACKEND=calculix` in the environment to activate v2. No code changes needed.

---

## Is it plug-and-play?

**Yes, at the simulator boundary.** The verifier, env, task schema, and tests are unchanged.

The only interface contract between the simulator and the rest of the system is:

```python
SimulationResult.outputs = {
    "safety_factor":     float,
    "max_deflection_mm": float,
    "max_stress_mpa":    float,
    "volume_mm3":        float,
    "mass_kg":           float,
    "bounding_box_mm":   list[float, float, float],
    "total_cost_usd":    float,
    ...
}
```

As long as `_run_fea` returns `{"max_stress_mpa", "safety_factor", "max_deflection_mm"}`,
everything downstream works identically.

---

## Migration strategy

| Phase | Action | When |
|-------|--------|------|
| Now | Ship v1 (beam theory) | ✅ Done |
| After first customers | Add `PyNiteFEA` as v1.5 option (no system deps, better than bounding-box) | 1–2 months |
| After RL gaming confirmed | Integrate CalculiX as v2 | 3–6 months |
| After v2 stable | Move CalculiX to Pro verifier; keep beam theory in basic | 6+ months |

The v1 → v1.5 upgrade (PyNiteFEA) takes about one day and requires no system dependencies.
The v1.5 → v2 upgrade (CalculiX) takes about one week including the gmsh meshing pipeline.

---

## References

- CalculiX: http://www.calculix.de — FEM solver, ABAQUS-compatible input
- gmsh: https://gmsh.info — mesh generation from STL
- PyNiteFEA: https://github.com/JWock82/PyNite — pure Python 3D structural analysis
- FEniCS: https://fenicsproject.org — Python-native FEM (more complex setup)

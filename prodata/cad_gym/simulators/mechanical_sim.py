"""
Mechanical simulator: executes CadQuery code and runs simplified FEA.

Agent code must define a variable `result` of type cadquery.Workplane.
"""

import tempfile
from pathlib import Path

from prodata.core.base_simulator import BaseSimulator, SimulationResult

# Material properties: yield strength, elastic modulus, density, raw material cost
# yield_mpa: yield strength (MPa)
# E_mpa:     Young's modulus (MPa)
# density:   kg/mm³
# cost_kg:   USD/kg raw material
MATERIAL_PROPS: dict[str, dict] = {
    "aluminum_6061_t6": {"yield_mpa": 276,  "E_mpa": 69_000,  "density": 2.70e-6, "cost_kg": 3.50},
    "aluminum_7075_t6": {"yield_mpa": 503,  "E_mpa": 71_700,  "density": 2.81e-6, "cost_kg": 8.00},
    "steel_mild":       {"yield_mpa": 250,  "E_mpa": 200_000, "density": 7.85e-6, "cost_kg": 2.00},
    "steel_316":        {"yield_mpa": 207,  "E_mpa": 193_000, "density": 7.98e-6, "cost_kg": 5.00},
    "steel_4140":       {"yield_mpa": 655,  "E_mpa": 205_000, "density": 7.85e-6, "cost_kg": 3.50},
    "titanium_6al4v":   {"yield_mpa": 880,  "E_mpa": 114_000, "density": 4.43e-6, "cost_kg": 35.00},
    "pla":              {"yield_mpa":  50,  "E_mpa":   3_500, "density": 1.24e-6, "cost_kg": 20.00},
    "petg":             {"yield_mpa":  50,  "E_mpa":   2_100, "density": 1.27e-6, "cost_kg": 25.00},
    "abs":              {"yield_mpa":  40,  "E_mpa":   2_300, "density": 1.05e-6, "cost_kg": 22.00},
}

PROCESS_COSTS: dict[str, float] = {
    "cnc_milling":        50.0,
    "3d_printing":         5.0,
    "laser_cut":          20.0,
    "sheet_metal_brake":  30.0,
    "wire_edm":           80.0,
    "casting":           100.0,
}

_DEFAULT_MATERIAL = MATERIAL_PROPS["aluminum_6061_t6"]


class MechanicalSimulator(BaseSimulator):
    """
    Executes CadQuery design code and returns geometry + FEA estimates.

    Uses beam-theory FEA (fast, deterministic). Replace _run_fea() with
    CalculiX or FEniCS for higher fidelity once RL baseline is established.
    """

    REQUIRED_VARIABLES = ["result"]
    ALLOWED_IMPORTS = ["cadquery"]

    def execute(self, code: str, task_spec: dict) -> SimulationResult:
        try:
            namespace = self._safe_exec(code, self.ALLOWED_IMPORTS)
        except RuntimeError as exc:
            return SimulationResult(success=False, error=str(exc))

        cad_object = namespace["result"]
        mat = self._material_props(task_spec)

        try:
            stl_path = self._export_stl(cad_object)
            geometry = self._analyze_geometry(stl_path, mat)
            fea = self._run_fea(geometry, task_spec, mat)
            cost = self._estimate_cost(geometry, task_spec, mat)
        except Exception as exc:
            return SimulationResult(success=False, error=str(exc))

        return SimulationResult(
            success=True,
            mesh_file=stl_path,
            outputs={
                "volume_mm3":          geometry["volume_mm3"],
                "mass_kg":             geometry["mass_kg"],
                "bounding_box_mm":     geometry["bounding_box_mm"],
                "max_stress_mpa":      fea["max_stress_mpa"],
                "safety_factor":       fea["safety_factor"],
                "max_deflection_mm":   fea["max_deflection_mm"],
                "material_cost_usd":   cost["material_cost_usd"],
                "fabrication_cost_usd": cost["fabrication_cost_usd"],
                "total_cost_usd":      cost["total_cost_usd"],
            },
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _material_props(self, task_spec: dict) -> dict:
        name = task_spec.get("material", "aluminum_6061_t6")
        return MATERIAL_PROPS.get(name, _DEFAULT_MATERIAL)

    def _export_stl(self, cad_object) -> str:
        import cadquery as cq
        tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        cq.exporters.export(cad_object, tmp.name)
        return tmp.name

    def _analyze_geometry(self, stl_path: str, mat: dict) -> dict:
        import trimesh
        mesh = trimesh.load(stl_path)
        bbox = mesh.bounds[1] - mesh.bounds[0]
        mass_kg = mesh.volume * mat["density"]
        return {
            "valid":           mesh.is_watertight,
            "volume_mm3":      float(mesh.volume),
            "mass_kg":         float(mass_kg),
            "bounding_box_mm": bbox.tolist(),
            "surface_area_mm2": float(mesh.area),
        }

    def _run_fea(self, geometry: dict, task_spec: dict, mat: dict) -> dict:
        """
        Cantilever beam-theory FEA with combined vertical + lateral loads.
        σ = M·c / I  (bending stress)
        δ = F·L³ / (3·E·I)  (tip deflection)
        """
        load_kg         = task_spec.get("load_kg", 10)
        lateral_load_kg = task_spec.get("lateral_load_kg", 0)
        extension_mm    = task_spec.get("extension_mm", 150)
        yield_mpa       = task_spec.get("yield_strength_mpa") or mat["yield_mpa"]
        E_mpa           = mat["E_mpa"]

        bbox = geometry["bounding_box_mm"]
        # Use largest horizontal extent as width, z-extent as depth (bending direction)
        b = max(bbox[0], bbox[1], 1.0)
        h = max(bbox[2], 1.0)

        I_mm4 = (b * h**3) / 12
        c_mm  = h / 2

        # Vertical load
        F_v   = load_kg * 9.81
        M_v   = F_v * extension_mm
        σ_v   = (M_v * c_mm) / I_mm4 if I_mm4 > 0 else 999.0

        # Lateral load (bending in orthogonal plane, b and h swap)
        I_lat  = (h * b**3) / 12
        F_lat  = lateral_load_kg * 9.81
        M_lat  = F_lat * extension_mm
        σ_lat  = (M_lat * (b / 2)) / I_lat if I_lat > 0 else 0.0

        # Combined (conservative: sum of bending stresses)
        stress_mpa    = σ_v + σ_lat
        safety_factor = yield_mpa / stress_mpa if stress_mpa > 0 else 0.0

        # Tip deflection (vertical — dominant)
        deflection_mm = (F_v * extension_mm**3) / (3 * E_mpa * I_mm4) if I_mm4 > 0 else 999.0

        return {
            "max_stress_mpa":    float(stress_mpa),
            "safety_factor":     float(safety_factor),
            "max_deflection_mm": float(deflection_mm),
        }

    def _estimate_cost(self, geometry: dict, task_spec: dict, mat: dict) -> dict:
        mass_kg  = geometry["mass_kg"]
        material = task_spec.get("material", "aluminum_6061_t6")
        process  = task_spec.get("process", "cnc_milling")

        mat_cost = mass_kg * MATERIAL_PROPS.get(material, _DEFAULT_MATERIAL)["cost_kg"]
        fab_cost = PROCESS_COSTS.get(process, 25.0)

        return {
            "material_cost_usd":    round(mat_cost, 2),
            "fabrication_cost_usd": round(fab_cost, 2),
            "total_cost_usd":       round(mat_cost + fab_cost, 2),
        }

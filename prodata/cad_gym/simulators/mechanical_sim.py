"""
Mechanical simulator: executes CadQuery code and runs simplified FEA.

Agent code must define a variable `result` of type cadquery.Workplane.
"""

import tempfile
from pathlib import Path

from prodata.core.base_simulator import BaseSimulator, SimulationResult


class MechanicalSimulator(BaseSimulator):
    """
    Executes CadQuery design code and returns geometry + FEA estimates.

    For production use, swap _run_fea() with CalculiX or FEniCS.
    Current implementation uses beam theory estimates (fast, good enough for RL).
    """

    REQUIRED_VARIABLES = ["result"]

    ALLOWED_IMPORTS = ["cadquery"]

    def execute(self, code: str, task_spec: dict) -> SimulationResult:
        try:
            namespace = self._safe_exec(code, self.ALLOWED_IMPORTS)
        except RuntimeError as exc:
            return SimulationResult(success=False, error=str(exc))

        cad_object = namespace["result"]

        try:
            stl_path = self._export_stl(cad_object)
            geometry = self._analyze_geometry(stl_path)
            fea = self._run_fea(geometry, task_spec)
            cost = self._estimate_cost(geometry, task_spec)
        except Exception as exc:
            return SimulationResult(success=False, error=str(exc))

        return SimulationResult(
            success=True,
            mesh_file=stl_path,
            outputs={
                # Geometry
                "volume_mm3": geometry["volume_mm3"],
                "mass_kg": geometry["mass_kg"],
                "bounding_box_mm": geometry["bounding_box_mm"],
                # FEA
                "max_stress_mpa": fea["max_stress_mpa"],
                "safety_factor": fea["safety_factor"],
                "max_deflection_mm": fea["max_deflection_mm"],
                # Cost
                "material_cost_usd": cost["material_cost_usd"],
                "fabrication_cost_usd": cost["fabrication_cost_usd"],
                "total_cost_usd": cost["total_cost_usd"],
            },
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _export_stl(self, cad_object) -> str:
        import cadquery as cq

        tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        cq.exporters.export(cad_object, tmp.name)
        return tmp.name

    def _analyze_geometry(self, stl_path: str) -> dict:
        import trimesh
        import numpy as np

        mesh = trimesh.load(stl_path)
        bbox = mesh.bounds[1] - mesh.bounds[0]  # [x, y, z] extents in mm

        # Assume aluminum 6061 density unless overridden
        density_kg_mm3 = 2.7e-6
        mass_kg = mesh.volume * density_kg_mm3

        return {
            "valid": mesh.is_watertight,
            "volume_mm3": float(mesh.volume),
            "mass_kg": float(mass_kg),
            "bounding_box_mm": bbox.tolist(),
            "surface_area_mm2": float(mesh.area),
        }

    def _run_fea(self, geometry: dict, task_spec: dict) -> dict:
        """
        Simplified beam-theory FEA estimate.
        Replace with CalculiX for higher accuracy.
        """
        import numpy as np

        load_kg = task_spec.get("load_kg", 10)
        extension_mm = task_spec.get("extension_mm", 150)
        yield_mpa = task_spec.get("yield_strength_mpa", 276)  # Al 6061-T6

        force_n = load_kg * 9.81
        moment_nmm = force_n * extension_mm

        bbox = geometry["bounding_box_mm"]
        h = max(bbox[2], 1.0)  # thickness dimension
        b = max(bbox[0], 1.0)  # width dimension

        # Second moment of area for rectangle
        I_mm4 = (b * h**3) / 12
        c_mm = h / 2

        # Bending stress σ = M*c / I
        stress_mpa = (moment_nmm * c_mm) / I_mm4 if I_mm4 > 0 else 999.0

        safety_factor = yield_mpa / stress_mpa if stress_mpa > 0 else 0.0

        # Simplified deflection δ = F*L³ / (3*E*I), E=69 GPa for Al
        E_mpa = 69000
        deflection_mm = (force_n * extension_mm**3) / (3 * E_mpa * I_mm4) if I_mm4 > 0 else 999.0

        return {
            "max_stress_mpa": float(stress_mpa),
            "safety_factor": float(safety_factor),
            "max_deflection_mm": float(deflection_mm),
        }

    def _estimate_cost(self, geometry: dict, task_spec: dict) -> dict:
        mass_kg = geometry["mass_kg"]

        material_costs = {
            "aluminum_6061_t6": 3.50,
            "steel_mild": 2.00,
            "steel_316": 5.00,
            "pla": 0.02,
        }
        process_costs = {
            "cnc_milling": 50.0,
            "3d_printing": 5.0,
            "laser_cut": 20.0,
        }

        material = task_spec.get("material", "aluminum_6061_t6")
        process = task_spec.get("process", "cnc_milling")

        mat_cost = mass_kg * material_costs.get(material, 3.0)
        fab_cost = process_costs.get(process, 25.0)

        return {
            "material_cost_usd": round(mat_cost, 2),
            "fabrication_cost_usd": round(fab_cost, 2),
            "total_cost_usd": round(mat_cost + fab_cost, 2),
        }

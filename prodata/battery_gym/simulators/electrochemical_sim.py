"""
Electrochemical simulator using PyBaMM.

Agent code must define `params` — a dict of cell design parameters.
The simulator runs PyBaMM SPM with lumped thermal model and returns:
  - energy_density_whkg   Wh/kg (gravimetric energy density)
  - capacity_ah           Ah discharge capacity
  - peak_temperature_c    °C peak cell temperature
  - cycle_life_80pct      estimated cycles to 80% capacity retention
  - estimated_cost_kwh    USD/kWh estimated cell cost
  - discharge_curve       dict with capacity_ah, voltage_v lists (for plotting)

Two modes:
  "live"    — real PyBaMM SPM simulation (~2–5 s/step, accurate)
  "dataset" — KDTree lookup in pre-computed dataset (~1 ms/step, approximate)

Agent action format:
    params = {
        "chemistry":                         "NMC532",   # NMC532 | LFP | NCA
        "negative_electrode_thickness":       75e-6,     # m
        "negative_electrode_porosity":        0.30,
        "negative_electrode_particle_radius": 5.5e-6,   # m
        "positive_electrode_thickness":       67e-6,     # m
        "positive_electrode_porosity":        0.335,
        "positive_electrode_particle_radius": 5.22e-6,  # m
        "separator_thickness":                25e-6,     # m
        "separator_porosity":                 0.47,
        "ambient_temperature_celsius":        25.0,
    }
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from prodata.core.base_simulator import BaseSimulator, SimulationResult


# ---------------------------------------------------------------------------
# Chemistry configuration
# ---------------------------------------------------------------------------

CHEMISTRY_CONFIG: dict[str, dict] = {
    "NMC532": {
        "param_set": "Chen2020",
        "discharge_cutoff_v": 2.5,
        "charge_cutoff_v": 4.2,
        "base_cycle_life": 1000,       # cycles to 80% at 1C/1C, 25°C
        "base_cost_kwh": 85.0,         # USD/kWh, 2024 cell-level
        "pos_am_density": 4700,        # kg/m³ NMC active material
    },
    "LFP": {
        "param_set": "Prada2013",
        "discharge_cutoff_v": 2.0,
        "charge_cutoff_v": 3.65,
        "base_cycle_life": 4000,
        "base_cost_kwh": 60.0,
        "pos_am_density": 3500,        # kg/m³ LFP active material
    },
    "NCA": {
        "param_set": "Chen2020",       # NMC used as proxy; similar electrochemistry
        "discharge_cutoff_v": 2.5,
        "charge_cutoff_v": 4.2,
        "base_cycle_life": 800,
        "base_cost_kwh": 95.0,
        "pos_am_density": 4600,        # kg/m³ NCA active material
    },
}

VALID_CHEMISTRIES = set(CHEMISTRY_CONFIG.keys())

# Physical electrode winding area for a 21700 cylindrical cell.
# Chen2020 parameter set: height=0.065 m, width=1.58 m → area ≈ 0.1027 m²
# (This is the full winding area used in PyBaMM's SPM model.)
ELECTRODE_AREA_M2 = 0.1027

# Material densities (kg/m³)
NEG_AM_DENSITY = 2200       # graphite active material
SEPARATOR_DENSITY = 900     # PE/PP separator
ELECTROLYTE_DENSITY = 1200  # LiPF6 in EC/DMC

# Fixed mass for components not modelled by electrode geometry:
# Cu anode current collector (~6g) + Al cathode CC (~3g)
# + steel casing + tabs + header (~10g) + excess electrolyte (~6g)
CELL_OVERHEAD_KG = 0.025

# Parameter bounds for validation and clamping
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "negative_electrode_thickness":       (20e-6,  200e-6),
    "negative_electrode_porosity":        (0.10,   0.65),
    "negative_electrode_particle_radius": (0.5e-6, 20e-6),
    "positive_electrode_thickness":       (20e-6,  180e-6),
    "positive_electrode_porosity":        (0.10,   0.65),
    "positive_electrode_particle_radius": (0.5e-6, 15e-6),
    "separator_thickness":                (10e-6,  60e-6),
    "separator_porosity":                 (0.25,   0.70),
    "ambient_temperature_celsius":        (-40.0,  60.0),
}

DEFAULTS: dict[str, float | str] = {
    "chemistry":                         "NMC532",
    "negative_electrode_thickness":       75e-6,
    "negative_electrode_porosity":        0.30,
    "negative_electrode_particle_radius": 5.5e-6,
    "positive_electrode_thickness":       67e-6,
    "positive_electrode_porosity":        0.335,
    "positive_electrode_particle_radius": 5.22e-6,
    "separator_thickness":                25e-6,
    "separator_porosity":                 0.47,
    "ambient_temperature_celsius":        25.0,
}

# Normalisation scales for KDTree (match PARAM_BOUNDS ranges)
_DATASET_PARAM_COLS = [
    "negative_electrode_thickness",
    "negative_electrode_porosity",
    "negative_electrode_particle_radius",
    "positive_electrode_thickness",
    "positive_electrode_porosity",
    "positive_electrode_particle_radius",
    "separator_thickness",
    "separator_porosity",
    "c_rate_discharge",
    "ambient_temperature_celsius",
]
_DATASET_SCALES = np.array([
    100e-6,   # neg thickness
    1.0,      # neg porosity
    10e-6,    # neg particle radius
    100e-6,   # pos thickness
    1.0,      # pos porosity
    10e-6,    # pos particle radius
    30e-6,    # sep thickness
    1.0,      # sep porosity
    2.0,      # c_rate
    50.0,     # temperature °C
])
_DATASET_OUTPUT_COLS = [
    "energy_density_whkg",
    "capacity_ah",
    "peak_temperature_c",
    "cycle_life_80pct",
    "estimated_cost_kwh",
]


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class ElectrochemicalSimulator(BaseSimulator):
    """
    Executes agent-designed cell parameters through PyBaMM electrochemical simulation.

    Agent code must define `params` (dict). The simulator validates, clamps
    out-of-range values, then runs PyBaMM SPM with lumped thermal model.
    """

    REQUIRED_VARIABLES = ["params"]

    def __init__(
        self,
        mode: str = "live",
        dataset_path: str | Path | None = None,
    ):
        """
        Args:
            mode:         "live" | "dataset"
            dataset_path: path to Parquet file (required when mode="dataset")
        """
        if mode not in ("live", "dataset"):
            raise ValueError(f"mode must be 'live' or 'dataset', got {mode!r}")
        self.mode = mode
        self._dataset: DatasetLookup | None = None

        if mode == "dataset":
            if dataset_path is None:
                raise ValueError("dataset_path is required when mode='dataset'")
            self._dataset = DatasetLookup(dataset_path)

    # ------------------------------------------------------------------
    # BaseSimulator interface
    # ------------------------------------------------------------------

    def execute(self, code: str, task_spec: dict) -> SimulationResult:
        # 1. Execute agent code
        try:
            namespace = self._safe_exec(code)
        except RuntimeError as exc:
            return SimulationResult(success=False, error=str(exc))

        if "params" not in namespace or not isinstance(namespace["params"], dict):
            return SimulationResult(
                success=False,
                error="Agent code must define `params` as a dict",
            )

        raw_params = namespace["params"]

        # 2. Validate + clamp
        params, warnings = self._validate_and_clamp(raw_params)

        # 3. Simulate
        try:
            if self.mode == "dataset":
                assert self._dataset is not None
                c_rate_discharge = float(task_spec.get("c_rate_discharge", 1.0))
                query_params = {**params, "c_rate_discharge": c_rate_discharge}
                outputs = self._dataset.query(query_params)
            else:
                outputs = self._run_pybamm(params, task_spec)
        except Exception as exc:
            return SimulationResult(success=False, error=str(exc))

        outputs["param_warnings"] = warnings
        return SimulationResult(success=True, outputs=outputs)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_and_clamp(self, params: dict) -> tuple[dict, list[str]]:
        warnings: list[str] = []
        result: dict = {**DEFAULTS}  # start from defaults, override below

        # Chemistry
        chem = params.get("chemistry", "NMC532")
        if chem not in VALID_CHEMISTRIES:
            warnings.append(f"Unknown chemistry '{chem}', using NMC532")
            chem = "NMC532"
        result["chemistry"] = chem

        # Numerical params
        for key, (lo, hi) in PARAM_BOUNDS.items():
            if key in params:
                try:
                    val = float(params[key])
                except (TypeError, ValueError):
                    warnings.append(f"{key}: expected float, got {params[key]!r}; using default")
                    continue
                if val < lo or val > hi:
                    warnings.append(
                        f"{key} = {val:.3e} is outside valid range [{lo:.3e}, {hi:.3e}]; clamped"
                    )
                result[key] = float(np.clip(val, lo, hi))

        # Physical consistency checks
        neg_active = (1 - result["negative_electrode_porosity"])
        if neg_active < 0.2:
            warnings.append("Negative electrode porosity > 0.8; unrealistically high void fraction")

        pos_active = (1 - result["positive_electrode_porosity"])
        if pos_active < 0.2:
            warnings.append("Positive electrode porosity > 0.8; unrealistically high void fraction")

        return result, warnings

    # ------------------------------------------------------------------
    # PyBaMM simulation
    # ------------------------------------------------------------------

    def _run_pybamm(self, params: dict, task_spec: dict) -> dict:
        import pybamm

        chemistry = params["chemistry"]
        cfg = CHEMISTRY_CONFIG[chemistry]

        # Load base parameter set
        param_values = pybamm.ParameterValues(cfg["param_set"])
        param_values = self._apply_agent_params(param_values, params)

        # C-rates from task spec
        c_rate_discharge = float(task_spec.get("c_rate_discharge", 1.0))
        c_rate_charge = float(task_spec.get("c_rate_charge", 1.0))

        cutoff = cfg["discharge_cutoff_v"]
        charge_v = cfg["charge_cutoff_v"]

        # Lumped thermal works for Chen2020 (NMC/NCA); Prada2013 (LFP) lacks the
        # current collector thermal parameters required by lumped thermal, so we
        # fall back to isothermal for LFP (peak temp reported as ambient).
        thermal_opt = "isothermal" if cfg["param_set"] == "Prada2013" else "lumped"
        model = pybamm.lithium_ion.SPM(options={"thermal": thermal_opt})

        experiment = pybamm.Experiment([
            f"Discharge at {c_rate_discharge}C until {cutoff} V",
            "Rest for 5 minutes",
            f"Charge at {c_rate_charge}C until {charge_v} V",
            "Rest for 5 minutes",
        ])

        sim = pybamm.Simulation(
            model,
            parameter_values=param_values,
            experiment=experiment,
        )

        try:
            solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
        except Exception:
            solver = pybamm.CasadiSolver(mode="safe")

        sol = sim.solve(solver=solver)

        # Extract outputs
        energy_density_whkg, capacity_ah = self._compute_energy_density(sol, params, chemistry)
        peak_temp_c = self._extract_peak_temperature(sol, params)
        cycle_life = self._estimate_cycle_life(params, chemistry, c_rate_charge, peak_temp_c)
        cost_kwh = self._estimate_cost(params, chemistry, energy_density_whkg)
        discharge_curve = self._extract_discharge_curve(sol)

        return {
            "energy_density_whkg": round(float(energy_density_whkg), 2),
            "capacity_ah":         round(float(capacity_ah), 4),
            "peak_temperature_c":  round(float(peak_temp_c), 2),
            "cycle_life_80pct":    int(cycle_life),
            "estimated_cost_kwh":  round(float(cost_kwh), 1),
            "discharge_curve":     discharge_curve,
        }

    def _apply_agent_params(self, param_values, params: dict):
        """Override PyBaMM parameter values with agent design choices."""
        overrides = {
            "Negative electrode thickness [m]":       "negative_electrode_thickness",
            "Negative electrode porosity":            "negative_electrode_porosity",
            "Negative particle radius [m]":           "negative_electrode_particle_radius",
            "Positive electrode thickness [m]":       "positive_electrode_thickness",
            "Positive electrode porosity":            "positive_electrode_porosity",
            "Separator thickness [m]":                "separator_thickness",
            "Separator porosity":                     "separator_porosity",
        }
        # LFP (Prada2013) uses ~50 nm nanoparticles for the positive electrode.
        # Overriding to µm-scale particles collapses capacity by 100×, so we skip
        # positive particle radius for LFP and let PyBaMM use its calibrated value.
        chem = params.get("chemistry", "NMC532")
        if chem != "LFP":
            overrides["Positive particle radius [m]"] = "positive_electrode_particle_radius"

        for pybamm_key, param_key in overrides.items():
            try:
                param_values[pybamm_key] = float(params[param_key])
            except (KeyError, Exception):
                pass  # Keep PyBaMM default if key not present or param set differs

        # Thermal boundary conditions
        ambient_k = float(params["ambient_temperature_celsius"]) + 273.15
        try:
            param_values["Ambient temperature [K]"] = ambient_k
            param_values["Initial temperature [K]"] = ambient_k
        except Exception:
            pass

        return param_values

    def _compute_energy_density(
        self, sol, params: dict, chemistry: str
    ) -> tuple[float, float]:
        """Discharge energy (Wh) / cell mass (kg) = Wh/kg."""
        # np.trapz was renamed to np.trapezoid in NumPy 2.0
        try:
            _trapz = np.trapezoid
        except AttributeError:
            _trapz = np.trapz  # type: ignore[attr-defined]
        try:
            discharge_step = sol.cycles[0].steps[0]
            time_s = discharge_step["Time [s]"].entries
            voltage_v = discharge_step["Terminal voltage [V]"].entries
            current_a = discharge_step["Current [A]"].entries
            capacity_ah = float(discharge_step["Discharge capacity [A.h]"].entries[-1])

            # Energy in Wh — current is positive during discharge in PyBaMM convention
            energy_wh = float(abs(_trapz(voltage_v * current_a, time_s)) / 3600.0)
        except Exception:
            # Fallback: get capacity from discharge step directly
            try:
                capacity_ah = float(sol.cycles[0].steps[0]["Discharge capacity [A.h]"].entries[-1])
            except Exception:
                capacity_ah = float(sol["Discharge capacity [A.h]"].entries.max())
            nominal_v = 3.2 if chemistry == "LFP" else 3.7
            energy_wh = capacity_ah * nominal_v

        cell_mass_kg = self._compute_cell_mass(params, chemistry)
        if cell_mass_kg <= 0:
            return 0.0, capacity_ah

        return energy_wh / cell_mass_kg, capacity_ah

    def _compute_cell_mass(self, params: dict, chemistry: str) -> float:
        """Estimate cell mass from electrode geometry (all SI units).

        Components:
          - Negative electrode active material (graphite)
          - Electrolyte filling negative electrode pores
          - Positive electrode active material (NMC / LFP / NCA)
          - Electrolyte filling positive electrode pores
          - Separator (polymer + electrolyte in pores)
          - Fixed overhead: current collectors, casing, tabs, excess electrolyte
        """
        cfg = CHEMISTRY_CONFIG[chemistry]
        area = ELECTRODE_AREA_M2

        neg_t = float(params["negative_electrode_thickness"])
        neg_p = float(params["negative_electrode_porosity"])
        pos_t = float(params["positive_electrode_thickness"])
        pos_p = float(params["positive_electrode_porosity"])
        sep_t = float(params["separator_thickness"])
        sep_p = float(params["separator_porosity"])

        # Active material masses
        neg_mass    = area * neg_t * (1.0 - neg_p) * NEG_AM_DENSITY
        pos_mass    = area * pos_t * (1.0 - pos_p) * cfg["pos_am_density"]

        # Electrolyte filling electrode pores
        neg_elyte   = area * neg_t * neg_p * ELECTROLYTE_DENSITY
        pos_elyte   = area * pos_t * pos_p * ELECTROLYTE_DENSITY

        # Separator (polymer skeleton + electrolyte in pores)
        sep_mass    = area * sep_t * (
            (1.0 - sep_p) * SEPARATOR_DENSITY + sep_p * ELECTROLYTE_DENSITY
        )

        return neg_mass + pos_mass + neg_elyte + pos_elyte + sep_mass + CELL_OVERHEAD_KG

    def _extract_peak_temperature(self, sol, params: dict) -> float:
        """Peak cell temperature during simulation in °C."""
        try:
            temp_k = sol["Cell temperature [K]"].entries
            return float(np.max(temp_k)) - 273.15
        except Exception:
            # Thermal model unavailable — report ambient
            return float(params["ambient_temperature_celsius"])

    def _estimate_cycle_life(
        self,
        params: dict,
        chemistry: str,
        c_rate_charge: float,
        peak_temp_c: float,
    ) -> int:
        """
        Physics-based cycle life estimate. Runs in milliseconds.

        Degradation mechanisms modelled:
          - SEI growth rate vs C-rate (Attia et al. 2022)
          - Graphite particle cracking vs particle radius (Zhao et al. 2011)
          - Restricted Li-ion transport at low porosity
          - Arrhenius temperature acceleration (Ea ≈ 25 kJ/mol)

        Returns estimated number of cycles to 80% capacity retention.
        """
        base = CHEMISTRY_CONFIG[chemistry]["base_cycle_life"]

        # --- C-rate factor: each C above 1C → faster SEI growth + plating risk ---
        # Empirical: ~30% fewer cycles per C above 1C (linear approximation)
        excess_c = max(0.0, c_rate_charge - 1.0)
        c_factor = 1.0 / (1.0 + 0.30 * excess_c)

        # --- Particle cracking: graphite particles >5 µm crack under volume change ---
        neg_r_um = float(params["negative_electrode_particle_radius"]) * 1e6
        particle_factor = max(0.5, 1.0 - 0.025 * max(0.0, neg_r_um - 5.0))

        # --- Transport factor: low porosity → restricted Li+ → localised lithiation ---
        neg_por = float(params["negative_electrode_porosity"])
        if neg_por >= 0.20:
            transport_factor = 1.0
        else:
            transport_factor = max(0.4, neg_por / 0.20)

        # --- Temperature Arrhenius factor (reference 25°C = 298.15 K) ---
        # Degradation rate: k_deg ∝ exp(-Ea / R·T)
        # Higher cell temperature → faster degradation → fewer cycles.
        # cycle_life ∝ 1/k_deg, so:
        #   temp_factor = k_deg_ref / k_deg_cell
        #               = exp(Ea/R · (1/T_cell − 1/T_ref))
        # At T_cell > T_ref: exponent < 0 → temp_factor < 1 (fewer cycles). ✓
        # Ea = 25 kJ/mol for graphite SEI growth (Attia et al. 2022)
        T_ref_k = 298.15
        T_cell_k = peak_temp_c + 273.15
        Ea_over_R = 3009.0  # K  (25000 J/mol / 8.314 J/mol/K)
        temp_factor = float(np.exp(Ea_over_R * (1.0 / T_cell_k - 1.0 / T_ref_k)))
        # Clamp: avoid runaway values at extreme temps
        temp_factor = float(np.clip(temp_factor, 0.1, 2.0))

        # --- Cold temperature: lithium plating at low temp (below 10°C) ---
        ambient_c = float(params["ambient_temperature_celsius"])
        if ambient_c < 10.0:
            plating_factor = max(0.3, 1.0 + 0.05 * (ambient_c - 10.0))
        else:
            plating_factor = 1.0

        cycle_life = int(
            base * c_factor * particle_factor * transport_factor
            * temp_factor * plating_factor
        )
        return int(np.clip(cycle_life, 50, 20_000))

    def _estimate_cost(
        self, params: dict, chemistry: str, energy_density_whkg: float
    ) -> float:
        """
        Estimate cell cost in USD/kWh.

        Scales the chemistry baseline cost by active material usage
        relative to the Chen2020 reference geometry.
        """
        base_cost = CHEMISTRY_CONFIG[chemistry]["base_cost_kwh"]

        # Thicker electrodes = more active material per unit energy
        # (cost/kWh decreases slightly with thicker electrodes because
        # overhead is amortised over more energy)
        ref_neg_t = 75e-6
        ref_pos_t = 67e-6
        neg_t = float(params["negative_electrode_thickness"])
        pos_t = float(params["positive_electrode_thickness"])
        thickness_ratio = (ref_neg_t + ref_pos_t) / (neg_t + pos_t + 1e-9)

        # More electrode material → overhead amortises better → lower $/kWh
        cost = base_cost * (0.8 + 0.2 * float(np.clip(thickness_ratio, 0.5, 2.0)))
        return round(float(cost), 1)

    def _extract_discharge_curve(self, sol) -> dict:
        """Voltage vs capacity during discharge (downsampled to ≤200 points)."""
        try:
            step = sol.cycles[0].steps[0]
            capacity = step["Discharge capacity [A.h]"].entries.tolist()
            voltage = step["Terminal voltage [V]"].entries.tolist()

            n = len(capacity)
            if n > 200:
                idx = np.linspace(0, n - 1, 200, dtype=int).tolist()
                capacity = [capacity[i] for i in idx]
                voltage = [voltage[i] for i in idx]

            return {"capacity_ah": capacity, "voltage_v": voltage}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Dataset lookup (for mode="dataset")
# ---------------------------------------------------------------------------

class DatasetLookup:
    """
    Fast nearest-neighbour lookup in a pre-computed PyBaMM simulation dataset.

    The dataset is a Parquet file generated by
    `prodata.battery_gym.scripts.precompute_dataset`.

    Builds a separate KDTree per chemistry for accuracy.
    Interpolates outputs with inverse-distance weighting (k=5 neighbours).
    """

    def __init__(self, path: str | Path):
        import pandas as pd
        from scipy.spatial import KDTree

        df = pd.read_parquet(path)

        self._trees: dict[str, dict] = {}
        for chem in VALID_CHEMISTRIES:
            sub = df[df["chemistry"] == chem].copy()
            if len(sub) == 0:
                continue
            features = sub[_DATASET_PARAM_COLS].values.astype(np.float64)
            targets = sub[_DATASET_OUTPUT_COLS].values.astype(np.float64)
            norm = features / _DATASET_SCALES
            self._trees[chem] = {
                "tree": KDTree(norm),
                "targets": targets,
            }

        if not self._trees:
            raise ValueError(f"Dataset at {path} contains no usable rows")

    def query(self, params: dict) -> dict:
        chemistry = str(params.get("chemistry", "NMC532"))
        if chemistry not in self._trees:
            chemistry = next(iter(self._trees))

        data = self._trees[chemistry]

        x = np.array([
            params.get("negative_electrode_thickness",       75e-6),
            params.get("negative_electrode_porosity",        0.30),
            params.get("negative_electrode_particle_radius", 5.5e-6),
            params.get("positive_electrode_thickness",       67e-6),
            params.get("positive_electrode_porosity",        0.335),
            params.get("positive_electrode_particle_radius", 5.22e-6),
            params.get("separator_thickness",                25e-6),
            params.get("separator_porosity",                 0.47),
            params.get("c_rate_discharge",                   1.0),
            params.get("ambient_temperature_celsius",        25.0),
        ], dtype=np.float64)

        x_norm = x / _DATASET_SCALES
        k = min(5, len(data["targets"]))
        dist, idx = data["tree"].query(x_norm, k=k)

        # Inverse-distance weighting
        weights = 1.0 / (dist + 1e-10)
        weights /= weights.sum()
        result = np.dot(weights, data["targets"][idx])

        return {
            col: float(val)
            for col, val in zip(_DATASET_OUTPUT_COLS, result)
        } | {"discharge_curve": {}}  # curves not stored in dataset

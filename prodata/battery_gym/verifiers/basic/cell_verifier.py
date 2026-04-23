"""
Basic battery cell verifier — open source, simple multi-dimensional checks.

Dimensions scored:
  - energy     (30%): gravimetric energy density vs target
  - cycle_life (35%): estimated cycles to 80% capacity retention
  - thermal    (20%): peak cell temperature during operation
  - cost       (15%): estimated USD/kWh vs budget

Intentionally does not detect reward hacking. Use Pro verifier for training.
"""

from prodata.core.base_verifier import BaseVerifier, VerificationResult
from prodata.core.base_simulator import SimulationResult
from prodata.core.task_schema import TaskSpec
from prodata.core.utils.scoring import weighted_score, linear_score, clamp
from prodata.battery_gym.task_schema import CellRequirements


class BasicCellVerifier(BaseVerifier):
    """
    Scores battery cell designs against task requirements.

    Pass condition:
      - overall_score >= 0.70
      - cycle_life score >= 0.40   (must not completely fail cycle life)
      - peak temperature <= max_peak_temp_c + 10°C headroom
    """

    WEIGHTS = {
        "energy":     0.30,
        "cycle_life": 0.35,
        "thermal":    0.20,
        "cost":       0.15,
    }

    def verify(self, sim_result: SimulationResult, task: TaskSpec) -> VerificationResult:
        if not sim_result.success:
            return VerificationResult(
                passed=False,
                overall_score=0.0,
                dimension_scores={k: 0.0 for k in self.WEIGHTS},
                warnings=[sim_result.error or "Simulation failed"],
            )

        out = sim_result.outputs
        req: CellRequirements = task.requirements

        scores = {
            "energy":     self._score_energy(out, req),
            "cycle_life": self._score_cycle_life(out, req),
            "thermal":    self._score_thermal(out, req),
            "cost":       self._score_cost(out, req),
        }

        overall = weighted_score(scores, self.WEIGHTS)

        passed = (
            overall >= 0.70
            and scores["cycle_life"] >= 0.40
            and out.get("peak_temperature_c", 999.0) <= req.max_peak_temp_c + 10.0
        )

        warnings = self._collect_warnings(out, req)
        if out.get("param_warnings"):
            warnings.extend(out["param_warnings"])

        return VerificationResult(
            passed=passed,
            overall_score=round(overall, 4),
            dimension_scores={k: round(v, 4) for k, v in scores.items()},
            gaming_detected=False,   # Basic verifier does not detect gaming
            warnings=warnings,
        )

    # ------------------------------------------------------------------

    def _score_energy(self, out: dict, req: CellRequirements) -> float:
        """
        Linear score from 0 to target energy density.
        Bonus (capped at 1.0) for exceeding target by up to 20%.
        """
        actual = out.get("energy_density_whkg", 0.0)
        target = req.target_energy_density_whkg
        if target <= 0:
            return 1.0
        # Linearly ramp from 0 at half target to 1.0 at target; flat above
        score = linear_score(actual, target=target, worst=target * 0.5)
        return clamp(score)

    def _score_cycle_life(self, out: dict, req: CellRequirements) -> float:
        """
        Linear score from 0 to min_cycle_life.
        80% of target → score = 0, target → score = 1.0.
        """
        actual = float(out.get("cycle_life_80pct", 0))
        target = float(req.min_cycle_life)
        if target <= 0:
            return 1.0
        score = linear_score(actual, target=target, worst=target * 0.5)
        return clamp(score)

    def _score_thermal(self, out: dict, req: CellRequirements) -> float:
        """
        1.0 when peak temperature is 5°C below limit.
        Linearly degrades to 0.0 at limit + 15°C (hard fail zone).
        """
        peak = out.get("peak_temperature_c", 999.0)
        limit = req.max_peak_temp_c
        if peak <= limit - 5.0:
            return 1.0
        if peak >= limit + 15.0:
            return 0.0
        return clamp(1.0 - (peak - (limit - 5.0)) / 20.0)

    def _score_cost(self, out: dict, req: CellRequirements) -> float:
        """
        1.0 when at or under budget.
        Linearly degrades above budget; 0 at 2× budget.
        """
        actual = out.get("estimated_cost_kwh", 999.0)
        budget = req.max_cost_kwh
        if budget <= 0:
            return 1.0
        if actual <= budget:
            return 1.0
        return clamp(1.0 - (actual - budget) / budget)

    def _collect_warnings(self, out: dict, req: CellRequirements) -> list[str]:
        warnings: list[str] = []

        energy = out.get("energy_density_whkg", 0.0)
        if energy < req.target_energy_density_whkg * 0.7:
            warnings.append(
                f"Energy density {energy:.1f} Wh/kg is far below "
                f"target {req.target_energy_density_whkg:.1f} Wh/kg"
            )

        cycles = out.get("cycle_life_80pct", 0)
        if cycles < req.min_cycle_life * 0.5:
            warnings.append(
                f"Estimated cycle life {cycles} is below 50% of "
                f"requirement {req.min_cycle_life}"
            )

        peak_t = out.get("peak_temperature_c", 0.0)
        if peak_t > req.max_peak_temp_c:
            warnings.append(
                f"Peak temperature {peak_t:.1f}°C exceeds limit {req.max_peak_temp_c:.1f}°C"
            )

        cost = out.get("estimated_cost_kwh", 0.0)
        if cost > req.max_cost_kwh:
            warnings.append(
                f"Estimated cost ${cost:.1f}/kWh exceeds budget ${req.max_cost_kwh:.1f}/kWh"
            )

        return warnings

"""
Basic bracket verifier — open source, simple checks.

Intentionally limited: checks key constraints but does NOT detect
reward hacking or multi-dimensional gaming. Use Pro verifier for training.
"""

from prodata.core.base_verifier import BaseVerifier, VerificationResult
from prodata.core.base_simulator import SimulationResult
from prodata.core.task_schema import TaskSpec
from prodata.core.utils.scoring import weighted_score, linear_score, threshold_score, clamp


class BasicBracketVerifier(BaseVerifier):
    """
    Verifies bracket designs against task requirements.

    Dimensions scored:
      - structural:     safety factor and deflection
      - cost:           total cost vs budget
      - geometry:       bounding box compliance
    """

    WEIGHTS = {
        "structural": 0.50,
        "cost": 0.30,
        "geometry": 0.20,
    }

    def verify(self, sim_result: SimulationResult, task: TaskSpec) -> VerificationResult:
        if not sim_result.success:
            return VerificationResult(
                passed=False,
                overall_score=0.0,
                dimension_scores={},
                warnings=[sim_result.error or "Simulation failed"],
            )

        out = sim_result.outputs
        req = task.requirements

        scores = {
            "structural": self._score_structural(out, req),
            "cost": self._score_cost(out, req),
            "geometry": self._score_geometry(out, req),
        }

        overall = weighted_score(scores, self.WEIGHTS)
        passed = (
            overall >= 0.7
            and scores["structural"] >= 0.5  # Must not be structurally failing
        )

        warnings = self._collect_warnings(out, req)

        return VerificationResult(
            passed=passed,
            overall_score=round(overall, 4),
            dimension_scores={k: round(v, 4) for k, v in scores.items()},
            gaming_detected=False,  # Basic verifier does not detect gaming
            warnings=warnings,
        )

    # ------------------------------------------------------------------

    def _score_structural(self, out: dict, req) -> float:
        sf = out.get("safety_factor", 0.0)
        defl = out.get("max_deflection_mm", 999.0)

        sf_score = linear_score(sf, target=3.0, worst=0.0)
        defl_limit = getattr(req, "max_deflection_mm", 5.0)
        defl_score = threshold_score(defl, threshold=defl_limit, higher_is_better=False)

        return clamp((sf_score + defl_score) / 2)

    def _score_cost(self, out: dict, req) -> float:
        total = out.get("total_cost_usd", 999.0)
        budget = getattr(req, "max_cost_usd", 50.0)
        if total <= 0:
            return 0.0
        return clamp(budget / total)  # 1.0 if at/under budget, degrades above

    def _score_geometry(self, out: dict, req) -> float:
        bbox = out.get("bounding_box_mm", [0, 0, 0])
        max_bbox = getattr(req, "max_bounding_box_mm", [300, 300, 50])

        fits = all(bbox[i] <= max_bbox[i] for i in range(min(len(bbox), len(max_bbox))))
        return 1.0 if fits else 0.0

    def _collect_warnings(self, out: dict, req) -> list[str]:
        warnings = []
        sf = out.get("safety_factor", 0)
        if sf < 1.5:
            warnings.append(f"Low safety factor: {sf:.2f} (design may fail under load)")
        if out.get("total_cost_usd", 0) > getattr(req, "max_cost_usd", 50):
            warnings.append(f"Over budget: ${out['total_cost_usd']:.2f}")
        return warnings

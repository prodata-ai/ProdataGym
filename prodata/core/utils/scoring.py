"""Shared scoring utilities used by all domain verifiers."""


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Compute weighted average. Missing keys in scores are skipped."""
    total_w = sum(weights[k] for k in weights if k in scores)
    if total_w == 0:
        return 0.0
    return sum(scores[k] * weights[k] for k in weights if k in scores) / total_w


def threshold_score(value: float, threshold: float, higher_is_better: bool = True) -> float:
    """1.0 if value passes threshold, 0.0 otherwise."""
    if higher_is_better:
        return 1.0 if value >= threshold else 0.0
    return 1.0 if value <= threshold else 0.0


def linear_score(
    value: float,
    target: float,
    worst: float,
) -> float:
    """
    Linear interpolation score.
    Returns 1.0 at target, 0.0 at worst, clamped.

    Example: safety_factor target=3.0, worst=0.0
      value=3.0 → 1.0
      value=1.5 → 0.5
      value=0.0 → 0.0
    """
    if target == worst:
        return 1.0 if value == target else 0.0
    return clamp((value - worst) / (target - worst))

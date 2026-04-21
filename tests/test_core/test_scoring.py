"""Unit tests for prodata.core.utils.scoring."""

import pytest
from prodata.core.utils.scoring import clamp, weighted_score, threshold_score, linear_score


class TestClamp:
    def test_within_range(self):
        assert clamp(0.5) == 0.5

    def test_below_min(self):
        assert clamp(-0.1) == 0.0

    def test_above_max(self):
        assert clamp(1.1) == 1.0

    def test_at_boundaries(self):
        assert clamp(0.0) == 0.0
        assert clamp(1.0) == 1.0

    def test_custom_range(self):
        assert clamp(5, lo=0, hi=10) == 5
        assert clamp(-1, lo=0, hi=10) == 0
        assert clamp(11, lo=0, hi=10) == 10


class TestWeightedScore:
    def test_equal_weights(self):
        scores = {"a": 1.0, "b": 0.0}
        weights = {"a": 0.5, "b": 0.5}
        assert weighted_score(scores, weights) == pytest.approx(0.5)

    def test_unequal_weights(self):
        scores = {"a": 1.0, "b": 0.0}
        weights = {"a": 0.8, "b": 0.2}
        assert weighted_score(scores, weights) == pytest.approx(0.8)

    def test_missing_score_key_is_skipped(self):
        # If a weight key has no corresponding score, skip it
        scores = {"a": 1.0}
        weights = {"a": 0.5, "b": 0.5}
        # Only "a" contributes; normalised by its weight only
        result = weighted_score(scores, weights)
        assert result == pytest.approx(1.0)

    def test_all_perfect(self):
        scores = {"structural": 1.0, "cost": 1.0, "geometry": 1.0}
        weights = {"structural": 0.5, "cost": 0.3, "geometry": 0.2}
        assert weighted_score(scores, weights) == pytest.approx(1.0)

    def test_all_zero(self):
        scores = {"structural": 0.0, "cost": 0.0, "geometry": 0.0}
        weights = {"structural": 0.5, "cost": 0.3, "geometry": 0.2}
        assert weighted_score(scores, weights) == pytest.approx(0.0)

    def test_empty_scores(self):
        assert weighted_score({}, {"a": 1.0}) == 0.0


class TestThresholdScore:
    def test_pass_higher_is_better(self):
        assert threshold_score(5.0, threshold=3.0, higher_is_better=True) == 1.0

    def test_fail_higher_is_better(self):
        assert threshold_score(2.0, threshold=3.0, higher_is_better=True) == 0.0

    def test_exactly_at_threshold(self):
        assert threshold_score(3.0, threshold=3.0, higher_is_better=True) == 1.0

    def test_pass_lower_is_better(self):
        assert threshold_score(1.0, threshold=3.0, higher_is_better=False) == 1.0

    def test_fail_lower_is_better(self):
        assert threshold_score(5.0, threshold=3.0, higher_is_better=False) == 0.0


class TestLinearScore:
    def test_at_target(self):
        assert linear_score(3.0, target=3.0, worst=0.0) == pytest.approx(1.0)

    def test_at_worst(self):
        assert linear_score(0.0, target=3.0, worst=0.0) == pytest.approx(0.0)

    def test_halfway(self):
        assert linear_score(1.5, target=3.0, worst=0.0) == pytest.approx(0.5)

    def test_beyond_target_clamped(self):
        # Value exceeding target should clamp to 1.0
        assert linear_score(5.0, target=3.0, worst=0.0) == pytest.approx(1.0)

    def test_below_worst_clamped(self):
        assert linear_score(-1.0, target=3.0, worst=0.0) == pytest.approx(0.0)

    def test_target_equals_worst(self):
        # Degenerate case: target == worst
        assert linear_score(1.0, target=2.0, worst=2.0) == 0.0
        assert linear_score(2.0, target=2.0, worst=2.0) == 1.0

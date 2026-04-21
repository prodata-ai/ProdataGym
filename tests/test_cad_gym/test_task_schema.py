"""Tests for BracketRequirements and BracketTaskSpec Pydantic schemas."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from prodata.cad_gym.task_schema import BracketRequirements, BracketTaskSpec
from prodata.core.task_schema import GradingCriteria

TASKS_FILE = Path(__file__).parent.parent.parent / "prodata/cad_gym/tasks/brackets_basic.json"


class TestBracketRequirements:
    def test_required_fields_present(self):
        req = BracketRequirements(load_kg=10, extension_mm=150)
        assert req.load_kg == 10.0
        assert req.extension_mm == 150.0

    def test_defaults_applied(self):
        req = BracketRequirements(load_kg=5, extension_mm=100)
        assert req.material == "aluminum_6061_t6"
        assert req.process == "cnc_milling"
        assert req.max_cost_usd == 50.0
        assert req.max_deflection_mm == 5.0
        assert req.lateral_load_kg == 0.0
        assert len(req.max_bounding_box_mm) == 3

    def test_missing_load_kg_raises(self):
        with pytest.raises(ValidationError):
            BracketRequirements(extension_mm=150)

    def test_missing_extension_mm_raises(self):
        with pytest.raises(ValidationError):
            BracketRequirements(load_kg=10)

    def test_custom_material(self):
        req = BracketRequirements(load_kg=5, extension_mm=100, material="steel_4140")
        assert req.material == "steel_4140"

    def test_lateral_load_optional(self):
        req = BracketRequirements(load_kg=10, extension_mm=150, lateral_load_kg=5)
        assert req.lateral_load_kg == 5.0

    def test_model_dump_includes_all_fields(self):
        req = BracketRequirements(load_kg=10, extension_mm=150)
        d = req.model_dump()
        assert "load_kg" in d
        assert "extension_mm" in d
        assert "material" in d
        assert "max_cost_usd" in d


class TestBracketTaskSpec:
    def _make_spec(self, **req_kwargs) -> BracketTaskSpec:
        return BracketTaskSpec(
            task_id="test_001",
            domain="mechanical",
            category="bracket",
            difficulty="easy",
            description="A test task",
            requirements=BracketRequirements(
                load_kg=req_kwargs.get("load_kg", 10),
                extension_mm=req_kwargs.get("extension_mm", 150),
            ),
            grading_criteria=GradingCriteria(
                weights={"structural": 0.5, "cost": 0.3, "geometry": 0.2}
            ),
        )

    def test_creates_successfully(self):
        spec = self._make_spec()
        assert spec.task_id == "test_001"
        assert spec.requirements.load_kg == 10.0

    def test_requirements_is_typed(self):
        spec = self._make_spec()
        # Should be BracketRequirements, not plain dict
        assert isinstance(spec.requirements, BracketRequirements)

    def test_difficulty_literal_valid(self):
        for d in ("easy", "medium", "hard"):
            spec = self._make_spec()
            spec = spec.model_copy(update={"difficulty": d})
            assert spec.difficulty == d

    def test_difficulty_literal_invalid(self):
        with pytest.raises(ValidationError):
            BracketTaskSpec(
                task_id="x",
                domain="mechanical",
                category="bracket",
                difficulty="extreme",  # invalid
                description="bad",
                requirements=BracketRequirements(load_kg=5, extension_mm=100),
                grading_criteria=GradingCriteria(weights={"structural": 1.0}),
            )


class TestTasksJsonFile:
    def test_file_exists(self):
        assert TASKS_FILE.exists(), f"Tasks file not found: {TASKS_FILE}"

    def test_file_has_50_tasks(self):
        with open(TASKS_FILE) as f:
            tasks = json.load(f)
        assert len(tasks) == 50, f"Expected 50 tasks, got {len(tasks)}"

    def test_all_tasks_parse_as_bracket_task_spec(self):
        with open(TASKS_FILE) as f:
            raw = json.load(f)
        errors = []
        for t in raw:
            try:
                BracketTaskSpec(**t)
            except Exception as e:
                errors.append(f"{t.get('task_id', '?')}: {e}")
        assert not errors, "Tasks failed validation:\n" + "\n".join(errors)

    def test_task_ids_are_unique(self):
        with open(TASKS_FILE) as f:
            raw = json.load(f)
        ids = [t["task_id"] for t in raw]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"

    def test_difficulty_distribution(self):
        with open(TASKS_FILE) as f:
            raw = json.load(f)
        counts = {"easy": 0, "medium": 0, "hard": 0}
        for t in raw:
            counts[t["difficulty"]] += 1
        assert counts["easy"] >= 10, f"Too few easy tasks: {counts}"
        assert counts["medium"] >= 15, f"Too few medium tasks: {counts}"
        assert counts["hard"] >= 8, f"Too few hard tasks: {counts}"

    def test_all_tasks_have_positive_loads(self):
        with open(TASKS_FILE) as f:
            raw = json.load(f)
        for t in raw:
            req = t["requirements"]
            assert req["load_kg"] > 0, f"{t['task_id']}: load_kg must be positive"
            assert req["extension_mm"] > 0, f"{t['task_id']}: extension_mm must be positive"

    def test_all_tasks_have_grading_weights_sum_to_one(self):
        with open(TASKS_FILE) as f:
            raw = json.load(f)
        for t in raw:
            weights = t["grading_criteria"]["weights"]
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, (
                f"{t['task_id']}: grading weights sum to {total}, not 1.0"
            )

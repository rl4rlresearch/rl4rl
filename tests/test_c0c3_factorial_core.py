# ruff: noqa: E402 -- the standalone experiments package is added explicitly.

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.c0c3_factorial.analysis import RunOutcome, estimate
from experiments.c0c3_factorial.environment import controlled_subprocess_environment
from experiments.c0c3_factorial.prompts import (
    PromptContext,
    PromptRenderer,
    VisibleCandidate,
    treatment_skeleton,
)
from experiments.c0c3_factorial.spec import (
    BudgetSpec,
    Condition,
    ExecutionBackend,
    FactorialSpec,
    FrameworkKind,
    FrameworkSpec,
    ModelSpec,
    ObjectiveDirection,
    TaskSpec,
    make_assignments,
)
from experiments.c0c3_factorial.state import (
    Candidate,
    Evaluation,
    SearchController,
    Usage,
)

TEMPLATES = ROOT / "experiments/c0c3_factorial/templates"


def protocol(*, proposals: int = 4, capacity: int = 2) -> FactorialSpec:
    return FactorialSpec(
        protocol_version="1.0",
        study_id="factorial-test",
        study_seed=20260820,
        blocks=2,
        portfolio_capacity=capacity,
        transition_opportunities=(2, 4) if proposals >= 4 else (2,),
        model=ModelSpec("gpt-test", "high"),
        budget=BudgetSpec(
            proposals=proposals,
            candidate_evaluations=proposals,
            max_total_tokens=100_000,
            max_evaluator_seconds=1000.0,
            evaluator_timeout_seconds=100,
        ),
    )


def task() -> TaskSpec:
    return TaskSpec(
        task_id="toy",
        display_name="Toy task",
        adapter="command_json_v1",
        seed_source="assets/toy.py",
        editable_paths=("candidate.py",),
        evaluator_command=("python", "evaluate.py"),
        objective_metric="score",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        qualification_metric=None,
        qualification_minimum=None,
        public_feedback_metrics=("score",),
        metric_patterns={"score": r"score:\s*([0-9.]+)"},
        final_holdout_command=("python", "holdout.py"),
        preferred_backend=ExecutionBackend.LOCAL,
    )


def framework() -> FrameworkSpec:
    return FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_editor_v1",
        prompt_profile="controlled_v1",
        diff_mode=True,
    )


def seed() -> Candidate:
    return Candidate(
        candidate_id="seed",
        parent_ids=[],
        fitness=0.0,
        metrics={"score": 0.0},
        artifact_path="candidates/seed",
        hypothesis="baseline",
        intended_edit="none",
        created_opportunity=0,
        retained_order=0,
    )


def context(condition: Condition, opportunity: int) -> PromptContext:
    visible = (VisibleCandidate("seed", 0.0, {"score": 0.0}, 0, "visible/slot-1"),)
    return PromptContext(
        condition=condition,
        opportunity=opportunity,
        selected_parent_id="seed",
        visible_candidates=visible,
        remaining_proposals=4,
        remaining_evaluations=4,
        remaining_tokens=100_000,
        remaining_evaluator_seconds=1000.0,
    )


def test_condition_mapping_and_frozen_transition_schedule() -> None:
    assert not Condition.C0.has_portfolio
    assert not Condition.C1.has_portfolio
    assert Condition.C2.has_portfolio
    assert Condition.C3.has_portfolio
    schedule = (2, 4)
    assert not Condition.C0.transition_active(2, schedule)
    assert Condition.C1.transition_active(2, schedule)
    assert not Condition.C2.transition_active(2, schedule)
    assert Condition.C3.transition_active(2, schedule)
    assert not Condition.C1.transition_active(3, schedule)


def test_block_randomization_is_balanced_deterministic_and_seed_paired() -> None:
    spec = protocol()
    first = make_assignments(spec, task_id="toy", framework_id="autoresearch")
    second = make_assignments(spec, task_id="toy", framework_id="autoresearch")
    assert first == second
    for block in (1, 2):
        rows = [row for row in first if row.block == block]
        assert {row.condition for row in rows} == set(Condition)
        assert len({row.run_seed for row in rows}) == 1
        assert sorted(row.order for row in rows) == [1, 2, 3, 4]


def test_prompts_share_one_skeleton_and_only_scheduled_cells_transition() -> None:
    renderer = PromptRenderer(TEMPLATES)
    spec = protocol()
    rendered = {
        condition: renderer.render(spec, task(), framework(), context(condition, 2))
        for condition in Condition
    }
    skeletons = {treatment_skeleton(value.text) for value in rendered.values()}
    assert len(skeletons) == 1
    assert not rendered[Condition.C0].transition_active
    assert rendered[Condition.C1].transition_active
    assert not rendered[Condition.C2].transition_active
    assert rendered[Condition.C3].transition_active
    assert (
        rendered[Condition.C0].proposal_policy_sha256
        == rendered[Condition.C2].proposal_policy_sha256
    )
    assert (
        rendered[Condition.C1].proposal_policy_sha256
        == rendered[Condition.C3].proposal_policy_sha256
    )


def test_single_incumbent_retains_only_strict_improvement(tmp_path: Path) -> None:
    controller = SearchController.create(
        tmp_path / "c0",
        protocol(),
        run_id="run-c0",
        condition=Condition.C0,
        seed_candidate=seed(),
    )
    active = controller.begin()
    assert active.visible_ids == ["seed"]
    rejected = controller.complete(
        candidate_id="worse",
        artifact_path="candidates/worse",
        hypothesis="worse",
        intended_edit="worse",
        evaluation=Evaluation(True, -1.0, {"score": -1.0}, 1.0),
        usage=Usage(input_tokens=10, output_tokens=2),
        prompt_hashes={"prompt": "x"},
    )
    assert not rejected["retained"]
    assert controller.state.portfolio_ids == ["seed"]
    controller.begin()
    accepted = controller.complete(
        candidate_id="better",
        artifact_path="candidates/better",
        hypothesis="better",
        intended_edit="better",
        evaluation=Evaluation(True, 1.0, {"score": 1.0}, 1.0),
        usage=Usage(input_tokens=10, output_tokens=2),
        prompt_hashes={"prompt": "y"},
    )
    assert accepted["retained"]
    assert controller.state.portfolio_ids == ["better"]
    assert controller.state.incumbent_id == "better"


def test_portfolio_fills_then_replaces_selected_parent(tmp_path: Path) -> None:
    controller = SearchController.create(
        tmp_path / "c2",
        protocol(capacity=2),
        run_id="run-c2",
        condition=Condition.C2,
        seed_candidate=seed(),
    )
    first = controller.begin()
    assert first.selected_parent_id == "seed"
    controller.complete(
        candidate_id="branch",
        artifact_path="candidates/branch",
        hypothesis="new branch",
        intended_edit="branch",
        evaluation=Evaluation(True, -1.0, {"score": -1.0}, 1.0),
        usage=Usage(input_tokens=5, output_tokens=1),
        prompt_hashes={},
    )
    assert controller.state.portfolio_ids == ["seed", "branch"]
    second = controller.begin()
    # branch has never been selected; seed was selected once.
    assert second.selected_parent_id == "branch"
    result = controller.complete(
        candidate_id="branch-better",
        artifact_path="candidates/branch-better",
        hypothesis="improve branch",
        intended_edit="change",
        evaluation=Evaluation(True, -0.5, {"score": -0.5}, 1.0),
        usage=Usage(input_tokens=5, output_tokens=1),
        prompt_hashes={},
    )
    assert result["retained"]
    assert result["evicted_candidate_id"] == "branch"
    assert controller.state.portfolio_ids == ["seed", "branch-better"]


def test_portfolio_fills_from_seed_and_replacement_preserves_lineage_count(
    tmp_path: Path,
) -> None:
    controller = SearchController.create(
        tmp_path / "c2-lineages",
        protocol(capacity=3),
        run_id="run-c2-lineages",
        condition=Condition.C2,
        seed_candidate=seed(),
    )
    for candidate_id, fitness in (("branch-a", -1.0), ("branch-b", -2.0)):
        active = controller.begin()
        assert active.selected_parent_id == "seed"
        controller.complete(
            candidate_id=candidate_id,
            artifact_path=f"candidates/{candidate_id}",
            hypothesis="independent initial branch",
            intended_edit="branch",
            evaluation=Evaluation(True, fitness, {"score": fitness}, 1.0),
            usage=Usage(input_tokens=1, output_tokens=1),
            prompt_hashes={},
        )
    selected = controller.begin()
    assert selected.selected_parent_id == "branch-a"
    controller.complete(
        candidate_id="branch-a-child",
        artifact_path="candidates/branch-a-child",
        hypothesis="improve branch a",
        intended_edit="change",
        evaluation=Evaluation(True, -0.5, {"score": -0.5}, 1.0),
        usage=Usage(input_tokens=1, output_tokens=1),
        prompt_hashes={},
    )
    assert controller.state.candidates["branch-a-child"].selected_count == 1
    # branch-b remains the least-selected lineage; the successful child does
    # not reset to zero and monopolize the next opportunity.
    assert controller.begin().selected_parent_id == "branch-b"


def test_controlled_environment_exposes_full_and_python_seeds() -> None:
    environment = controlled_subprocess_environment(2**40 + 17)
    assert environment["C0C3_RUN_SEED"] == str(2**40 + 17)
    assert environment["PYTHONHASHSEED"] == "17"


def test_no_search_always_uses_seed_and_never_adapts(tmp_path: Path) -> None:
    controller = SearchController.create(
        tmp_path / "n0",
        protocol(),
        run_id="run-n0",
        condition=Condition.C3,
        seed_candidate=seed(),
        no_search=True,
    )
    for index in range(2):
        active = controller.begin()
        assert active.visible_ids == ["seed"]
        assert active.selected_parent_id == "seed"
        assert not active.transition_active
        record = controller.complete(
            candidate_id=f"independent-{index}",
            artifact_path=f"candidates/{index}",
            hypothesis="independent",
            intended_edit="change",
            evaluation=Evaluation(True, 100.0, {"score": 100.0}, 1.0),
            usage=Usage(input_tokens=1, output_tokens=1),
            prompt_hashes={},
        )
        assert record["retention_decision"] == "independent_not_retained"
    assert controller.state.incumbent_id == "seed"
    assert controller.state.portfolio_ids == ["seed"]


def test_event_log_has_required_accounting_fields(tmp_path: Path) -> None:
    controller = SearchController.create(
        tmp_path / "c1",
        protocol(),
        run_id="run-c1",
        condition=Condition.C1,
        seed_candidate=seed(),
    )
    controller.begin()
    controller.complete(
        candidate_id="invalid",
        artifact_path="candidates/invalid",
        hypothesis="test",
        intended_edit="break intentionally",
        evaluation=Evaluation(
            False, None, {"error": "syntax"}, 2.5, failure_kind="implementation"
        ),
        usage=Usage(
            input_tokens=20,
            cached_input_tokens=8,
            output_tokens=3,
            reasoning_output_tokens=1,
        ),
        prompt_hashes={"prompt_sha256": "abc"},
    )
    completed = [
        json.loads(line)
        for line in controller.events_path.read_text().splitlines()
        if json.loads(line)["event"] == "proposal_completed"
    ][0]
    required = {
        "condition",
        "visible_candidate_ids",
        "selected_parent_ids",
        "proposal_type",
        "hypothesis",
        "intended_edit",
        "candidate_id",
        "parent_ids",
        "evaluation",
        "retained",
        "retention_decision",
        "usage_increment",
        "evaluator_calls_increment",
        "evaluator_seconds_increment",
        "remaining_budget",
        "prompt_hashes",
    }
    assert required <= completed.keys()
    assert completed["usage_increment"]["total_tokens"] == 23


def test_factorial_estimators() -> None:
    rows = []
    values = {Condition.C0: 1, Condition.C1: 3, Condition.C2: 4, Condition.C3: 10}
    for block in ("b1", "b2"):
        rows.extend(
            RunOutcome(block, condition, value, f"{block}-{condition.value}")
            for condition, value in values.items()
        )
    result = estimate(rows)
    assert result["portfolio_memory_main_effect"] == pytest.approx(5.0)
    assert result["assumption_changing_main_effect"] == pytest.approx(4.0)
    assert result["interaction"] == pytest.approx(4.0)

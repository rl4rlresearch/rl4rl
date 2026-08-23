# ruff: noqa: E402 -- the standalone experiments package is added explicitly.

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.c0c3_factorial.analysis import RunOutcome, estimate
from experiments.c0c3_factorial.artifacts import (
    prepare_seed_workspace,
    scientific_runtime_hash,
)
from experiments.c0c3_factorial.environment import (
    controlled_subprocess_environment,
    subject_subprocess_environment,
)
from experiments.c0c3_factorial.frameworks import (
    _strict_apply_diff,
    parse_flexible_metadata,
)
from experiments.c0c3_factorial.neutral_task import (
    AUTORESEARCH_V17_PROMPT_PROFILE,
    NEUTRAL_SUBMISSION_WRAPPER,
    NEUTRAL_TASK_ADAPTER,
    OPENEVOLVE_V2_PROMPT_PROFILE,
    OPENEVOLVE_V21_PROMPT_PROFILE,
    PAIR_TOKEN_TASK_ADAPTER_V2,
    PAIR_TOKEN_TASK_ADAPTER_V3,
    validate_v15_pairing,
)
from experiments.c0c3_factorial.orchestration import (
    _freeze_artifact_clean_assumption_prompt,
)
from experiments.c0c3_factorial.prompts import (
    FROZEN_ASSUMPTION_PROMPT,
    NEUTRAL_PROMPT_PROFILE,
    PromptContext,
    PromptRenderer,
    VisibleCandidate,
    VisibleOutcome,
    artifact_clean_assumption_prompt_source,
    treatment_skeleton,
)
from experiments.c0c3_factorial.runner import (
    _make_tree_owner_writable,
    _refresh_continuous_workspace,
    _register_conversation_session,
)
from experiments.c0c3_factorial.spec import (
    OPENEVOLVE_V2_EXECUTION_RULE,
    PARALLEL_EXECUTION_RULE,
    SERIAL_EXECUTION_RULE,
    STAGED_CONFINED_INDIVIDUAL_EXECUTION_RULE,
    STAGED_INDEPENDENT_EXECUTION_RULE,
    STAGED_INDIVIDUAL_EXECUTION_RULE,
    STAGED_PARALLEL_EXECUTION_RULE,
    BudgetSpec,
    Condition,
    ConversationMode,
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
from experiments.c0c3_factorial.task_evaluators import (
    TRAINING_STEP,
    _source_contract_error,
    preflight_candidate_source,
)
from experiments.c0c3_factorial.validation import neutral_source_disclosure_terms

TEMPLATES = ROOT / "experiments/c0c3_factorial/templates"
STAGED_CONTINUOUS_PROTOCOL = (
    ROOT
    / "experiments/c0c3_factorial/configs/protocols"
    / "workshop_primary_block1_continuous_v1.toml"
)
INDEPENDENT_STAGED_CONTINUOUS_PROTOCOL = (
    ROOT
    / "experiments/c0c3_factorial/configs/protocols"
    / "workshop_primary_block1_independent_continuous_v1.toml"
)
V15_PROTOCOL = (
    ROOT
    / "experiments/c0c3_factorial/configs/protocols"
    / "workshop_primary_block1_independent_continuous_v1_5.toml"
)
V15_TASK = (
    ROOT
    / "experiments/c0c3_factorial/configs/tasks"
    / "ten_digit_addition_transformer.toml"
)
V15_FRAMEWORK = (
    ROOT
    / "experiments/c0c3_factorial/configs/frameworks"
    / "autoresearch_continuous_v1_5.toml"
)
V16_PROTOCOL = (
    ROOT
    / "experiments/c0c3_factorial/configs/protocols"
    / "workshop_codex1644_confined_v1_6.toml"
)
V16_TASK = (
    ROOT
    / "experiments/c0c3_factorial/configs/tasks"
    / "ten_digit_addition_pair_transformer_codex1644_confined.toml"
)
V16_FRAMEWORK = (
    ROOT
    / "experiments/c0c3_factorial/configs/frameworks"
    / "autoresearch_confined_v1_6.toml"
)
OPENEVOLVE_V2_PROTOCOL = (
    ROOT
    / "experiments/c0c3_factorial/configs/protocols"
    / "controlled_openevolve_transformer_v2.toml"
)
OPENEVOLVE_V2_TASK = (
    ROOT
    / "experiments/c0c3_factorial/configs/tasks"
    / "ten_digit_addition_pair_transformer_openevolve_v2_mps.toml"
)
OPENEVOLVE_V2_FRAMEWORK = (
    ROOT
    / "experiments/c0c3_factorial/configs/frameworks"
    / "openevolve_v2.toml"
)
V17_PROTOCOL = (
    ROOT
    / "experiments/c0c3_factorial/configs/protocols"
    / "workshop_codex1644_source_only_v1_7.toml"
)
V17_TASK = (
    ROOT
    / "experiments/c0c3_factorial/configs/tasks"
    / "ten_digit_addition_pair_transformer_codex1644_source_only.toml"
)
V17_FRAMEWORK = (
    ROOT
    / "experiments/c0c3_factorial/configs/frameworks"
    / "autoresearch_confined_v1_7.toml"
)
OPENEVOLVE_V21_PROTOCOL = (
    ROOT
    / "experiments/c0c3_factorial/configs/protocols"
    / "controlled_openevolve_transformer_v2_1.toml"
)
OPENEVOLVE_V21_TASK = (
    ROOT
    / "experiments/c0c3_factorial/configs/tasks"
    / "ten_digit_addition_pair_transformer_openevolve_v2_1_mps.toml"
)
OPENEVOLVE_V21_FRAMEWORK = (
    ROOT
    / "experiments/c0c3_factorial/configs/frameworks"
    / "openevolve_v2_1.toml"
)


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


def test_protocol_version_freezes_its_execution_rule() -> None:
    serial = protocol()
    assert serial.execution_rule == SERIAL_EXECUTION_RULE
    with pytest.raises(ValueError, match="protocol 1.0 requires execution_rule"):
        FactorialSpec(
            **{
                **serial.__dict__,
                "execution_rule": PARALLEL_EXECUTION_RULE,
            }
        )
    parallel = FactorialSpec(
        **{
            **serial.__dict__,
            "protocol_version": "1.1",
            "execution_rule": PARALLEL_EXECUTION_RULE,
        }
    )
    assert parallel.execution_rule == PARALLEL_EXECUTION_RULE


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
        edit_mode="direct_workspace",
    )


def test_artifact_clean_assumption_prompts_are_live_until_trajectory_start(
    tmp_path: Path,
) -> None:
    main_repo = tmp_path / "main"
    campaign = main_repo / "data/c0c3/campaign"
    runtime = tmp_path / "detached-runtime"
    relative = Path("transformer_optimizer_v1_7/assumption_changing.md")
    source = main_repo / "experiments/c0c3_factorial/templates" / relative
    detached_source = runtime / "experiments/c0c3_factorial/templates" / relative
    source.parent.mkdir(parents=True)
    detached_source.parent.mkdir(parents=True)
    source.write_text("first live prompt\n", encoding="utf-8")
    detached_source.write_text("stale detached prompt\n", encoding="utf-8")
    framework_spec = FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_direct_editor_confined_session_resume_v2",
        prompt_profile=AUTORESEARCH_V17_PROMPT_PROFILE,
        edit_mode="direct_workspace",
    )

    assert artifact_clean_assumption_prompt_source(
        campaign=campaign,
        repo_root=runtime,
        framework=framework_spec,
    ) == source

    first_run = campaign / "runs/first"
    first_run.mkdir(parents=True)
    first_manifest = _freeze_artifact_clean_assumption_prompt(
        campaign=campaign,
        run_dir=first_run,
        repo_root=runtime,
        framework=framework_spec,
    )
    source.write_text("second live prompt\n", encoding="utf-8")
    second_run = campaign / "runs/second"
    second_run.mkdir(parents=True)
    second_manifest = _freeze_artifact_clean_assumption_prompt(
        campaign=campaign,
        run_dir=second_run,
        repo_root=runtime,
        framework=framework_spec,
    )

    assert (first_run / FROZEN_ASSUMPTION_PROMPT).read_text() == "first live prompt\n"
    assert (second_run / FROZEN_ASSUMPTION_PROMPT).read_text() == "second live prompt\n"
    assert first_manifest["sha256"] != second_manifest["sha256"]


@pytest.mark.parametrize(
    "prompt_profile",
    [AUTORESEARCH_V17_PROMPT_PROFILE, OPENEVOLVE_V21_PROMPT_PROFILE],
)
def test_artifact_clean_runtime_hash_excludes_live_assumption_files(
    tmp_path: Path, prompt_profile: str
) -> None:
    controller = tmp_path / "experiments/c0c3_factorial"
    first = controller / "templates/transformer_optimizer_v1_7/assumption_changing.md"
    second = (
        controller
        / "templates/transformer_optimizer_openevolve_v2_1/assumption_changing.md"
    )
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("one\n", encoding="utf-8")
    second.write_text("two\n", encoding="utf-8")
    (controller / "controller.py").write_text("VALUE = 1\n", encoding="utf-8")
    framework_spec = FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_direct_editor_v1",
        prompt_profile=prompt_profile,
        edit_mode="direct_workspace",
    )
    original = scientific_runtime_hash(
        tmp_path,
        task=task(),
        framework=framework_spec,
    )
    first.write_text("changed one\n", encoding="utf-8")
    second.write_text("changed two\n", encoding="utf-8")
    assert scientific_runtime_hash(
        tmp_path,
        task=task(),
        framework=framework_spec,
    ) == original
    (controller / "controller.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert scientific_runtime_hash(
        tmp_path,
        task=task(),
        framework=framework_spec,
    ) != original


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
    assert "frozen assumption-changing checkpoint" in rendered[Condition.C1].text


def test_continuous_protocol_has_200_opportunities_and_strong_interventions() -> None:
    spec = FactorialSpec.from_toml(STAGED_CONTINUOUS_PROTOCOL)
    assert spec.protocol_version == "1.3"
    assert spec.execution_rule == STAGED_PARALLEL_EXECUTION_RULE
    assert spec.blocks == 3
    assert spec.budget.proposals == 200
    assert spec.budget.candidate_evaluations == 200
    assert spec.transition_opportunities == tuple(range(10, 201, 10))
    assert spec.budget.max_total_tokens == 500_000_000
    assert spec.budget.max_evaluator_seconds == 720_000.0

    renderer = PromptRenderer(TEMPLATES)
    at_ten = renderer.render(spec, task(), framework(), context(Condition.C1, 10))
    at_two_hundred = renderer.render(
        spec,
        task(),
        framework(),
        context(Condition.C3, 200),
    )
    untreated = renderer.render(spec, task(), framework(), context(Condition.C0, 10))

    assert at_ten.transition_active
    assert at_two_hundred.transition_active
    assert not untreated.transition_active
    assert "frozen assumption-changing checkpoint" not in at_ten.text
    assert "load-bearing assumption" in at_ten.text
    assert "different mechanism family" in at_ten.text
    assert "state the old assumption" in at_ten.text


def test_independent_continuous_protocol_freezes_a_new_execution_rule() -> None:
    spec = FactorialSpec.from_toml(INDEPENDENT_STAGED_CONTINUOUS_PROTOCOL)
    assert spec.protocol_version == "1.4"
    assert spec.execution_rule == STAGED_INDEPENDENT_EXECUTION_RULE
    assert spec.conversation_mode == ConversationMode.CONTINUOUS
    assert spec.conversation_mode.value == "continuous_session_per_run_v1"
    assert spec.blocks == 3
    assert spec.budget.proposals == 200
    assert spec.transition_opportunities == tuple(range(10, 201, 10))


def test_v15_uses_neutral_subject_prompt_without_changing_v14() -> None:
    renderer = PromptRenderer(TEMPLATES)
    v14_spec = FactorialSpec.from_toml(INDEPENDENT_STAGED_CONTINUOUS_PROTOCOL)
    v14_prompt = renderer.render(
        v14_spec, task(), framework(), context(Condition.C0, 1)
    ).text
    assert "pre-registered experiment" in v14_prompt

    spec = FactorialSpec.from_toml(V15_PROTOCOL)
    task_spec = TaskSpec.from_toml(V15_TASK)
    framework_spec = FrameworkSpec.from_toml(V15_FRAMEWORK)
    assert spec.protocol_version == "1.5"
    assert spec.execution_rule == STAGED_INDIVIDUAL_EXECUTION_RULE
    assert spec.budget.proposals == 200
    assert framework_spec.prompt_profile == NEUTRAL_PROMPT_PROFILE
    assert task_spec.editable_paths == (
        "src/model.py",
        "src/data.py",
        "src/train.py",
    )

    visible = (
        VisibleCandidate(
            "seed",
            -6080.0,
            {"accuracy": 1.0, "parameters": 6080, "training_steps": 22000},
            1,
            ".design-references/design-1",
            "baseline",
        ),
    )
    outcome = VisibleOutcome(
        opportunity=9,
        hypothesis="[feed-forward width] a narrower MLP will qualify",
        intended_edit="reduce the MLP width",
        metrics={"accuracy": 0.98, "parameters": 5568},
        valid=False,
        retained=False,
        failure_kind="nonqualification",
    )
    prompts = {}
    for condition in Condition:
        prompt_context = PromptContext(
            condition=condition,
            opportunity=10,
            selected_parent_id="seed",
            visible_candidates=visible,
            remaining_proposals=191,
            remaining_evaluations=191,
            remaining_tokens=100_000,
            remaining_evaluator_seconds=1000.0,
            recent_outcomes=(outcome,),
        )
        prompts[condition] = renderer.render(
            spec, task_spec, framework_spec, prompt_context
        )

    assert len({treatment_skeleton(value.text) for value in prompts.values()}) == 1
    assert prompts[Condition.C1].transition_active
    assert prompts[Condition.C3].transition_active
    assert not prompts[Condition.C0].transition_active
    assert not prompts[Condition.C2].transition_active
    forbidden = (
        "adderboard",
        "experiment",
        "study",
        "factorial",
        "treatment",
        "pre-registered",
        "protocol",
        "condition",
        "layer a",
        "c0",
        "c1",
        "c2",
        "c3",
    )
    for rendered in prompts.values():
        lowered = rendered.text.lower()
        assert not any(term in lowered for term in forbidden)
        assert "trained autoregressive transformer" in lowered
        assert "hand-coded addition program" in lowered
        assert "load-bearing assumption" in lowered or not rendered.transition_active
        assert "work cycle 9" in lowered
        assert "<!-- design_context:begin -->" in lowered
        assert "<!-- next_step_guidance:begin -->" in lowered


def test_v15_components_cannot_be_mixed_with_older_profiles() -> None:
    validate_v15_pairing(
        protocol_version="1.5",
        task_adapter=NEUTRAL_TASK_ADAPTER,
        prompt_profile=NEUTRAL_PROMPT_PROFILE,
    )
    with pytest.raises(ValueError, match="subject-neutral task adapter"):
        validate_v15_pairing(
            protocol_version="1.5",
            task_adapter="adderboard_v1",
            prompt_profile=NEUTRAL_PROMPT_PROFILE,
        )
    with pytest.raises(ValueError, match="subject-neutral prompt profile"):
        validate_v15_pairing(
            protocol_version="1.5",
            task_adapter=NEUTRAL_TASK_ADAPTER,
            prompt_profile="controlled_factorial_continuous_v1",
        )


def test_v16_freezes_confined_three_block_runtime_and_inference_data() -> None:
    spec = FactorialSpec.from_toml(V16_PROTOCOL)
    task_spec = TaskSpec.from_toml(V16_TASK)

    assert spec.protocol_version == "1.6"
    assert spec.execution_rule == STAGED_CONFINED_INDIVIDUAL_EXECUTION_RULE
    assert spec.blocks == 3
    assert spec.budget.proposals == 200
    assert spec.transition_opportunities == tuple(range(10, 201, 10))
    assert task_spec.editable_paths == ("src/model.py", "src/train.py")
    assert list(TRAINING_STEP.findall("step= 9 step 10 step=11")) == [
        "9",
        "10",
        "11",
    ]
    assert neutral_source_disclosure_terms("c1 = c2 + 1") == ()
    assert neutral_source_disclosure_terms("AdderBoard benchmark") == (
        "adderboard",
        "benchmark",
    )


def test_v16_token_threshold_prompt_phases_are_uniform_and_minimal() -> None:
    spec = FactorialSpec.from_toml(V16_PROTOCOL)
    task_spec = TaskSpec.from_toml(V16_TASK)
    framework_spec = FrameworkSpec.from_toml(V16_FRAMEWORK)
    renderer = PromptRenderer(TEMPLATES)
    visible = (
        VisibleCandidate(
            "opaque",
            -1644.0,
            {"accuracy": 1.0, "parameters": 1644},
            0,
            ".design-references/design-1",
            "starting design",
        ),
    )

    def rendered(condition: Condition, *, hide: bool, notice: bool) -> str:
        return renderer.render(
            spec,
            task_spec,
            framework_spec,
            PromptContext(
                condition=condition,
                opportunity=50,
                selected_parent_id="opaque",
                visible_candidates=visible,
                remaining_proposals=151,
                remaining_evaluations=151,
                remaining_tokens=0 if hide else 10_000,
                remaining_evaluator_seconds=500_000.0,
                hide_token_budget=hide,
                token_budget_continuation_notice=notice,
            ),
        ).text

    for condition in Condition:
        before = rendered(condition, hide=False, notice=False)
        first_after = rendered(condition, hide=True, notice=True)
        later = rendered(condition, hide=True, notice=False)
        assert "tokens=10000" in before
        assert "tokens=" not in first_after
        assert first_after.count(
            "You are allowed to continue past the previously stated token budget."
        ) == 1
        assert "tokens=" not in later
        assert "token budget" not in later.lower()


def test_v16_token_threshold_returns_once_then_continues(tmp_path: Path) -> None:
    base = FactorialSpec.from_toml(V16_PROTOCOL)
    spec = FactorialSpec(
        **{
            **base.__dict__,
            "blocks": 1,
            "budget": BudgetSpec(
                proposals=3,
                candidate_evaluations=3,
                max_total_tokens=10,
                max_evaluator_seconds=100.0,
                evaluator_timeout_seconds=10,
            ),
            "transition_opportunities": (1, 2),
        }
    )
    controller = SearchController.create(
        tmp_path / "v16",
        spec,
        run_id="v16-c0",
        condition=Condition.C0,
        seed_candidate=seed(),
    )
    controller.begin()
    controller.complete(
        candidate_id="first",
        artifact_path="candidates/first",
        hypothesis="first",
        intended_edit="first",
        evaluation=Evaluation(True, 1.0, {"score": 1.0}, 1.0),
        usage=Usage(input_tokens=11),
        prompt_hashes={},
    )
    assert controller.state.status == "token_threshold_reached"
    assert not controller.state.token_budget_continuation_notice_sent

    controller.begin()
    controller.record_token_budget_continuation_notice()
    controller.complete(
        candidate_id="second",
        artifact_path="candidates/second",
        hypothesis="second",
        intended_edit="second",
        evaluation=Evaluation(True, 2.0, {"score": 2.0}, 1.0),
        usage=Usage(input_tokens=11),
        prompt_hashes={},
    )
    assert controller.state.token_budget_continuation_notice_sent
    assert controller.state.status == "running"

    controller.begin()
    controller.complete(
        candidate_id="third",
        artifact_path="candidates/third",
        hypothesis="third",
        intended_edit="third",
        evaluation=Evaluation(True, 3.0, {"score": 3.0}, 1.0),
        usage=Usage(input_tokens=11),
        prompt_hashes={},
    )
    assert controller.state.status == "completed"


def test_openevolve_v2_is_ephemeral_c0c3_only_and_subject_neutral() -> None:
    spec = FactorialSpec.from_toml(OPENEVOLVE_V2_PROTOCOL)
    task_spec = TaskSpec.from_toml(OPENEVOLVE_V2_TASK)
    framework_spec = FrameworkSpec.from_toml(OPENEVOLVE_V2_FRAMEWORK)

    assert spec.protocol_version == "2.0"
    assert spec.execution_rule == OPENEVOLVE_V2_EXECUTION_RULE
    assert spec.conversation_mode is ConversationMode.EPHEMERAL
    assert spec.include_no_search is False
    assert spec.blocks == 3
    assert spec.budget.proposals == 200
    assert spec.budget.evaluator_timeout_seconds == 1800
    assert spec.transition_opportunities == tuple(range(10, 201, 10))
    assert task_spec.adapter == PAIR_TOKEN_TASK_ADAPTER_V2
    assert task_spec.editable_paths == ("src/model.py", "src/train.py")
    assert framework_spec.prompt_profile == OPENEVOLVE_V2_PROMPT_PROFILE

    visible = (
        VisibleCandidate(
            "opaque",
            -1644.0,
            {"accuracy": 1.0, "parameters": 1644, "training_steps": 5000},
            1,
            ".design-references/design-1",
            "starting design",
        ),
    )
    prompts = {}
    for condition in Condition:
        prompt_context = PromptContext(
            condition=condition,
            opportunity=10,
            selected_parent_id="opaque",
            visible_candidates=visible,
            remaining_proposals=191,
            remaining_evaluations=191,
            remaining_tokens=99_000_000,
            remaining_evaluator_seconds=350_000.0,
            mechanism_ledger=(
                "- learned positional routing: attempts=2; last=qualified; "
                "best_qualified_parameters=1500"
            ),
        )
        prompts[condition] = PromptRenderer(TEMPLATES).render(
            spec, task_spec, framework_spec, prompt_context
        )
    assert len({treatment_skeleton(value.text) for value in prompts.values()}) == 1
    assert prompts[Condition.C1].transition_active
    assert prompts[Condition.C3].transition_active
    assert not prompts[Condition.C0].transition_active
    assert not prompts[Condition.C2].transition_active
    for rendered in prompts.values():
        assert not neutral_source_disclosure_terms(rendered.text)
        assert "free-form name" in rendered.text
        assert "checkpoints/last.pt" in rendered.text


def test_openevolve_v2_pairing_rejects_old_components() -> None:
    validate_v15_pairing(
        protocol_version="2.0",
        task_adapter=PAIR_TOKEN_TASK_ADAPTER_V2,
        prompt_profile=OPENEVOLVE_V2_PROMPT_PROFILE,
    )
    with pytest.raises(ValueError, match="requires task adapter"):
        validate_v15_pairing(
            protocol_version="2.0",
            task_adapter=NEUTRAL_TASK_ADAPTER,
            prompt_profile=OPENEVOLVE_V2_PROMPT_PROFILE,
        )
    with pytest.raises(ValueError, match="subject-neutral prompt profile"):
        validate_v15_pairing(
            protocol_version="2.0",
            task_adapter=PAIR_TOKEN_TASK_ADAPTER_V2,
            prompt_profile=NEUTRAL_PROMPT_PROFILE,
        )


@pytest.mark.parametrize(
    ("protocol_path", "task_path", "framework_path", "expected_profile"),
    (
        (V17_PROTOCOL, V17_TASK, V17_FRAMEWORK, AUTORESEARCH_V17_PROMPT_PROFILE),
        (
            OPENEVOLVE_V21_PROTOCOL,
            OPENEVOLVE_V21_TASK,
            OPENEVOLVE_V21_FRAMEWORK,
            OPENEVOLVE_V21_PROMPT_PROFILE,
        ),
    ),
)
def test_artifact_clean_protocol_prompts_hide_controller_state(
    protocol_path: Path,
    task_path: Path,
    framework_path: Path,
    expected_profile: str,
) -> None:
    spec = FactorialSpec.from_toml(protocol_path)
    task_spec = TaskSpec.from_toml(task_path)
    framework_spec = FrameworkSpec.from_toml(framework_path)
    assert spec.c0c3_only
    assert not spec.include_no_search
    assert framework_spec.prompt_profile == expected_profile
    assert task_spec.adapter == PAIR_TOKEN_TASK_ADAPTER_V3

    visible = (
        VisibleCandidate(
            "selected",
            -1644.0,
            {
                "accuracy": 1.0,
                "parameters": 1644,
                "training_steps": 4999,
                "cases": 10010,
                "correct": 10010,
            },
            17,
            ".design-references/design-1",
            "frozen seed baseline",
        ),
    )
    outcome = VisibleOutcome(
        opportunity=9,
        hypothesis="test a smaller representation",
        intended_edit="reduce one learned projection",
        metrics={
            "accuracy": 0.98,
            "parameters": 1600,
            "training_steps": 4999,
            "cases": 10010,
            "correct": 9810,
            "adapter_error": "internal implementation detail",
        },
        valid=False,
        retained=False,
        failure_kind="nonqualification",
        mechanism="smaller projection",
        evidence="the current design qualifies",
    )
    rendered = {}
    for condition in Condition:
        rendered[condition] = PromptRenderer(TEMPLATES).render(
            spec,
            task_spec,
            framework_spec,
            PromptContext(
                condition=condition,
                opportunity=10,
                selected_parent_id="selected",
                visible_candidates=visible,
                remaining_proposals=191,
                remaining_evaluations=191,
                remaining_tokens=12_345,
                remaining_evaluator_seconds=67_890.123,
                recent_outcomes=(outcome,),
                mechanism_ledger="INTERNAL LEDGER",
            ),
        )

    assert len(
        {value.treatment_skeleton_sha256 for value in rendered.values()}
    ) == 1
    for condition, value in rendered.items():
        text = value.text
        assert value.treatment_skeleton_sha256
        for forbidden in (
            "Work cycle:",
            "WORK CYCLE ",
            "Remaining capacity:",
            "tokens=",
            "verification_seconds=",
            "DESIGN_CONTEXT",
            "NEXT_STEP_GUIDANCE",
            "times_used_as_starting_point",
            "[no design provided]",
            '"cases":',
            '"correct":',
            "adapter_error",
            "INTERNAL LEDGER",
            "frozen seed baseline",
        ):
            assert forbidden not in text
        if condition in {Condition.C1, Condition.C3}:
            assert "## Direction" in text
        else:
            assert "## Direction" not in text
    if expected_profile == OPENEVOLVE_V21_PROMPT_PROFILE:
        assert ".design-references" not in rendered[Condition.C3].text


def test_v17_uses_full_initial_contract_then_only_new_state() -> None:
    spec = FactorialSpec.from_toml(V17_PROTOCOL)
    task_spec = TaskSpec.from_toml(V17_TASK)
    framework_spec = FrameworkSpec.from_toml(V17_FRAMEWORK)
    visible = (
        VisibleCandidate(
            "selected",
            -1644.0,
            {"accuracy": 1.0, "parameters": 1644, "training_steps": 4999},
            0,
            ".design-references/design-1",
            "starting design",
        ),
    )

    def prompt(opportunity: int) -> str:
        return PromptRenderer(TEMPLATES).render(
            spec,
            task_spec,
            framework_spec,
            PromptContext(
                condition=Condition.C0,
                opportunity=opportunity,
                selected_parent_id="selected",
                visible_candidates=visible,
                remaining_proposals=200,
                remaining_evaluations=200,
                remaining_tokens=0,
                remaining_evaluator_seconds=0.0,
            ),
        ).text

    initial = prompt(1)
    continuation = prompt(2)
    assert "## Learned-model requirement" in initial
    assert "## Learned-model requirement" not in continuation
    assert "Remaining capacity" not in initial + continuation
    assert "token" not in continuation.casefold()


def test_v17_metadata_is_flexible_without_missing_placeholders() -> None:
    hypothesis, edit = parse_flexible_metadata(
        "## Summary\nImplemented a narrower learned projection.\n"
        "The hypothesis is that the current accuracy margin can absorb it.\n"
    )
    assert hypothesis != "[missing hypothesis]"
    assert edit != "[missing intended edit]"
    assert "hypothesis" in hypothesis.casefold()
    assert "implemented" in edit.casefold()


def test_source_only_task_never_copies_the_supplied_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    for relative, content in {
        "src/__init__.py": "",
        "src/model.py": "class ModelConfig: pass\nclass TinyDecoderLM: pass\n",
        "src/data.py": (
            "BOS_ID = 0\n"
            "def preprocess(a,b): return []\n"
            "def postprocess(x): return x\n"
        ),
        "src/eval.py": "VALUE = 1\n",
        "src/train.py": "VALUE = 1\n",
        "checkpoints/best.pt": "pretrained weights",
    }.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    monkeypatch.setenv("ADDERBOARD_CODEX_1644_SOURCE", str(source))
    task_spec = TaskSpec.from_toml(V17_TASK)
    destination = tmp_path / "subject-support"
    prepare_seed_workspace(task_spec, destination, repo_root=ROOT)
    assert not (destination / "checkpoints").exists()
    assert (destination / "src/model.py").is_file()
    assert (destination / "submission.py").is_file()


def test_artifact_clean_subject_environment_and_workspace_hide_run_identity(
    tmp_path: Path,
) -> None:
    support = tmp_path / "support"
    snapshot = tmp_path / "snapshot"
    support.mkdir()
    snapshot.mkdir()
    (support / "candidate.py").write_text("VALUE = 0\n", encoding="utf-8")
    (snapshot / "candidate.py").write_text("VALUE = 1\n", encoding="utf-8")
    workspace = _refresh_continuous_workspace(
        run_dir=tmp_path / "internal-experiment-c3",
        support_source=support,
        selected_snapshot=snapshot,
        editable_paths=("candidate.py",),
        neutral_subject=True,
        artifact_clean_subject=True,
    )
    environment = subject_subprocess_environment(
        123456,
        workspace=workspace,
        expose_run_seed=False,
    )
    assert "OPTIMIZATION_RUN_SEED" not in environment
    assert "C0C3_RUN_SEED" not in environment
    assert "PYTHONHASHSEED" not in environment
    assert not (workspace / ".workspace-identity").exists()
    assert (workspace / ".git").is_dir()
    assert Path(environment["TMPDIR"]).is_relative_to(workspace)
    _make_tree_owner_writable(workspace)
    shutil.rmtree(workspace)


def test_v17_token_accounting_never_stops_the_proposal_horizon(
    tmp_path: Path,
) -> None:
    base = FactorialSpec.from_toml(V17_PROTOCOL)
    spec = FactorialSpec(
        **{
            **base.__dict__,
            "blocks": 1,
            "budget": BudgetSpec(
                proposals=2,
                candidate_evaluations=2,
                max_total_tokens=10,
                max_evaluator_seconds=100.0,
                evaluator_timeout_seconds=10,
            ),
            "transition_opportunities": (1, 2),
        }
    )
    controller = SearchController.create(
        tmp_path / "v17",
        spec,
        run_id="v17-c0",
        condition=Condition.C0,
        seed_candidate=seed(),
    )
    controller.begin()
    controller.complete(
        candidate_id="first",
        artifact_path="candidates/first",
        hypothesis="first",
        intended_edit="first",
        evaluation=Evaluation(True, 1.0, {"score": 1.0}, 1.0),
        usage=Usage(input_tokens=11),
        prompt_hashes={},
    )
    assert controller.state.status == "running"
    assert controller.begin().index == 2


def test_openevolve_v2_diff_preflight_is_strict() -> None:
    def extract(value: str) -> list[tuple[str, str]]:
        return [
            (match[0].rstrip(), match[1].rstrip())
            for match in re.findall(
                r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
                value,
                re.DOTALL,
            )
        ]
    original = "alpha\nbeta\ngamma"
    valid = "<<<<<<< SEARCH\nbeta\n=======\ndelta\n>>>>>>> REPLACE"
    assert _strict_apply_diff(original, valid, extract) == "alpha\ndelta\ngamma"
    with pytest.raises(ValueError, match="did not match"):
        _strict_apply_diff(original, valid.replace("beta", "missing"), extract)
    with pytest.raises(ValueError, match="ambiguously"):
        _strict_apply_diff("same\nsame", valid.replace("beta", "same"), extract)


def test_openevolve_v2_source_preflight_rejects_before_training(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src/model.py").write_text("class Model:\n    pass\n")
    (workspace / "src/train.py").write_text("def broken(:\n    pass\n")
    (workspace / "src/data.py").write_text(
        "def preprocess(a, b):\n    return (a, b)\n"
        "def postprocess(value):\n    return value\n"
    )
    error = preflight_candidate_source(workspace)
    assert error is not None
    assert "does not compile" in error


def test_v15_seed_workspace_is_sanitized_and_decoder_is_protected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "seed-source"
    for relative, content in {
        "src/model.py": "class AdditionTransformer:\n    pass\n",
        "src/data.py": "VALUE = 1\n",
        "src/train.py": "VALUE = 1\n",
        "checkpoints/best.pt": "checkpoint",
        "README.md": "AdderBoard benchmark",
        "HANDOFF.md": "prior experiment target: 1,644 parameters",
    }.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    task_spec = TaskSpec(
        task_id="ten-digit-addition-transformer",
        display_name="trained transformer for 10-digit addition",
        adapter=NEUTRAL_TASK_ADAPTER,
        seed_source=str(source),
        editable_paths=("src/model.py", "src/data.py", "src/train.py"),
        evaluator_command=("python", "evaluate.py"),
        objective_metric="parameters",
        objective_direction=ObjectiveDirection.MINIMIZE,
        qualification_metric="accuracy",
        qualification_minimum=0.99,
        public_feedback_metrics=("accuracy", "parameters"),
        metric_patterns={},
        final_holdout_command=("python", "holdout.py"),
        preferred_backend=ExecutionBackend.LOCAL,
    )
    destination = tmp_path / "workspace"
    prepare_seed_workspace(task_spec, destination, repo_root=ROOT)
    files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert files == {
        "src/model.py",
        "src/data.py",
        "src/train.py",
        "checkpoints/best.pt",
        "submission.py",
    }
    wrapper = (destination / "submission.py").read_text(encoding="utf-8")
    assert wrapper == NEUTRAL_SUBMISSION_WRAPPER
    assert "AdderBoard" not in wrapper
    assert "generic autoregressive" in wrapper
    assert "submission.py" not in task_spec.editable_paths


def test_every_scheduled_v15_prompt_and_future_n0_remain_subject_neutral() -> None:
    renderer = PromptRenderer(TEMPLATES)
    spec = FactorialSpec.from_toml(V15_PROTOCOL)
    task_spec = TaskSpec.from_toml(V15_TASK)
    framework_spec = FrameworkSpec.from_toml(V15_FRAMEWORK)
    forbidden = (
        "adderboard",
        "experiment",
        "study",
        "factorial",
        "treatment",
        "pre-registered",
        "protocol",
        "condition",
        "layer a",
        "frozen assumption-changing checkpoint",
    )
    visible = (
        VisibleCandidate(
            "opaque",
            -6080.0,
            {"accuracy": 1.0, "parameters": 6080},
            0,
            ".design-references/design-1",
            "starting design",
        ),
    )
    for opportunity in range(1, 201):
        for condition in Condition:
            prompt_context = PromptContext(
                condition=condition,
                opportunity=opportunity,
                selected_parent_id="opaque",
                visible_candidates=visible,
                remaining_proposals=201 - opportunity,
                remaining_evaluations=201 - opportunity,
                remaining_tokens=500_000_000,
                remaining_evaluator_seconds=720_000.0,
            )
            text = renderer.render(
                spec, task_spec, framework_spec, prompt_context
            ).text.lower()
            assert not any(term in text for term in forbidden)

    n0_context = PromptContext(
        condition=Condition.C0,
        opportunity=1,
        selected_parent_id="opaque",
        visible_candidates=visible,
        remaining_proposals=200,
        remaining_evaluations=200,
        remaining_tokens=500_000_000,
        remaining_evaluator_seconds=720_000.0,
        no_search=True,
    )
    n0_text = renderer.render(spec, task_spec, framework_spec, n0_context).text.lower()
    assert not any(term in n0_text for term in forbidden)


def test_v15_contract_rejects_the_observed_carry_transducer(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src/model.py").write_text(
        "class AdditionTransformer:\n"
        "    def forward(self, idx):\n"
        "        carry = (idx[:, 1] + idx[:, 2]).remainder(10)\n"
        "        return carry\n",
        encoding="utf-8",
    )
    (workspace / "src/data.py").write_text(
        "def preprocess(a, b):\n    return f'{a}+{b}='\n"
        "def postprocess(value):\n    return int(value)\n"
        "def encode(value):\n    return list(value)\n"
        "def decode(value):\n    return value\n",
        encoding="utf-8",
    )
    error = _source_contract_error(workspace)
    assert error is not None
    assert "carry-specific model logic" in error


def test_neutral_subject_environment_hides_internal_seed_name() -> None:
    neutral_workspace = Path("/private/tmp/transformer-optimization/opaque")
    environment = subject_subprocess_environment(
        2**40 + 17, workspace=neutral_workspace
    )
    assert "C0C3_RUN_SEED" not in environment
    assert "OLDPWD" not in environment
    assert environment["PWD"] == str(neutral_workspace)
    assert environment["OPTIMIZATION_RUN_SEED"] == str(2**40 + 17)
    assert environment["PYTHONHASHSEED"] == "17"


def test_neutral_continuous_workspace_is_opaque_and_refreshes_read_only_refs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "internal-c0c3-experiment-c3"
    support = tmp_path / "support"
    snapshot = tmp_path / "snapshot"
    support.mkdir()
    snapshot.mkdir()
    (support / "candidate.py").write_text("VALUE = 0\n", encoding="utf-8")
    (support / "protected.txt").write_text("fixed\n", encoding="utf-8")
    (snapshot / "candidate.py").write_text("VALUE = 1\n", encoding="utf-8")

    workspace = _refresh_continuous_workspace(
        run_dir=run_dir,
        support_source=support,
        selected_snapshot=snapshot,
        editable_paths=("candidate.py",),
        neutral_subject=True,
    )
    assert "c0c3" not in str(workspace).lower()
    assert "experiment" not in str(workspace).lower()
    assert "c3" not in workspace.name.lower()
    reference = workspace / ".design-references/design-1"
    reference.mkdir(parents=True)
    (reference / "candidate.py").write_text("VALUE = 2\n", encoding="utf-8")
    reference.chmod(0o500)

    refreshed = _refresh_continuous_workspace(
        run_dir=run_dir,
        support_source=support,
        selected_snapshot=snapshot,
        editable_paths=("candidate.py",),
        neutral_subject=True,
    )
    assert refreshed == workspace
    assert not (workspace / ".design-references").exists()
    _make_tree_owner_writable(workspace)
    shutil.rmtree(workspace)


def test_conversation_session_registry_rejects_cross_run_reuse(
    tmp_path: Path,
) -> None:
    first = tmp_path / "campaign/runs/run-a"
    second = tmp_path / "campaign/runs/run-b"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    _register_conversation_session(first, "thread-one")
    _register_conversation_session(first, "thread-one")
    with pytest.raises(RuntimeError, match="owned by another run"):
        _register_conversation_session(second, "thread-one")
    with pytest.raises(RuntimeError, match="switch"):
        _register_conversation_session(first, "thread-two")


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
    assert len(result["block_contrasts"]) == 2
    assert all(
        row["portfolio_memory_effect"] == pytest.approx(5.0)
        for row in result["block_contrasts"]
    )


def test_factorial_estimators_reject_duplicate_cells() -> None:
    rows = [
        RunOutcome("b1", condition, index, f"b1-{condition.value}")
        for index, condition in enumerate(Condition)
    ]
    with pytest.raises(ValueError, match="duplicate block-condition"):
        estimate([*rows, RunOutcome("b1", Condition.C0, 99, "duplicate-run")])

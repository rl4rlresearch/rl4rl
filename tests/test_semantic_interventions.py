# ruff: noqa: E402 -- repository root is inserted for standalone test runs.

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.c0c3_factorial.campaign import calibrate_task
from experiments.c0c3_factorial.semantic_interventions import (
    create_semantic_campaign,
    run_semantic_opportunity,
    semantic_status,
    set_semantic_run_control,
    validate_semantic_campaign,
)
from experiments.c0c3_factorial.spec import (
    UNIFIED_V3_EXECUTION_RULE,
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
)
from experiments.c0c3_factorial.state import Candidate, Evaluation, SearchController
from experiments.c0c3_factorial.training_ladder import (
    assess_developmental_value,
    evaluate_training_ladder,
)


def _spec() -> FactorialSpec:
    return FactorialSpec(
        protocol_version="3.0",
        study_id="semantic-test",
        study_seed=44,
        blocks=1,
        portfolio_capacity=2,
        transition_opportunities=(2, 3),
        conversation_mode=ConversationMode.BOUNDED,
        model=ModelSpec("gpt-fake", "high"),
        budget=BudgetSpec(3, 3, 1_000_000, 100.0, 10),
        execution_rule=UNIFIED_V3_EXECUTION_RULE,
        include_no_search=False,
    )


def _phased_spec() -> FactorialSpec:
    return FactorialSpec(
        protocol_version="3.0",
        study_id="semantic-phased-test",
        study_seed=45,
        blocks=1,
        portfolio_capacity=2,
        transition_opportunities=(6,),
        conversation_mode=ConversationMode.BOUNDED,
        model=ModelSpec("gpt-fake", "high"),
        budget=BudgetSpec(6, 6, 1_000_000, 100.0, 10),
        execution_rule=UNIFIED_V3_EXECUTION_RULE,
        include_no_search=False,
    )


def _task(seed: Path, *, qualification: bool = False) -> TaskSpec:
    command = (
        "{python}",
        "evaluate.py",
        "--training-examples",
        "100000",
        "--output",
        "{output}",
    )
    return TaskSpec(
        task_id="semantic-toy",
        display_name="Toy optimization",
        adapter="command_json_v1",
        seed_source=str(seed),
        editable_paths=("candidate.py",),
        evaluator_command=command,
        objective_metric="score",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        qualification_metric="accuracy" if qualification else None,
        qualification_minimum=0.99 if qualification else None,
        public_feedback_metrics=("score", "accuracy"),
        metric_patterns={},
        final_holdout_command=command,
        preferred_backend=ExecutionBackend.LOCAL,
    )


def _seed(path: Path) -> Path:
    path.mkdir()
    (path / "candidate.py").write_text("SCORE = 0\n", encoding="utf-8")
    (path / "evaluate.py").write_text(
        """import argparse, json
from candidate import SCORE
p=argparse.ArgumentParser()
p.add_argument('--training-examples')
p.add_argument('--output', required=True)
a=p.parse_args()
open(a.output, 'w').write(json.dumps({'metrics': {'score': SCORE, 'accuracy': SCORE}}))
""",
        encoding="utf-8",
    )
    return path


def _plan(path: Path) -> Path:
    path.write_text(
        """schema_version = "4.0"
replicates = 1
shared_prefix_opportunities = 1
session_span_opportunities = 1
max_parallel_agent_calls = 2
task_evaluator_capacity = 1

[[interventions]]
id = "passive_control"
label = "Passive"
family = "control"
prompt_path = "passive_control.md"
components = []

[[interventions]]
id = "assumption_challenge"
label = "Assumption"
family = "epistemic"
prompt_path = "assumption_challenge.md"
components = ["assumption_challenge"]
""",
        encoding="utf-8",
    )
    return path


def _codex(path: Path, counter: Path) -> Path:
    path.write_text(
        f"""#!{sys.executable}
import json, pathlib, sys
args=sys.argv[1:]
workspace=pathlib.Path(args[args.index('--cd')+1])
last=pathlib.Path(args[args.index('--output-last-message')+1])
prompt=sys.stdin.read()
candidate=workspace/'candidate.py'
score=int(candidate.read_text().split('=')[1])
candidate.write_text(f'SCORE = {{score+1}}\\n')
last.write_text('HYPOTHESIS: increase score\\nINTENDED_EDIT: increment\\n')
counter=pathlib.Path({str(counter)!r})
counter.write_text(str(int(counter.read_text())+1 if counter.exists() else 1))
print(json.dumps({{'type':'thread.started','thread_id':'semantic'}}))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':10,'cached_input_tokens':1,'output_tokens':2,'reasoning_output_tokens':1}}}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _openevolve_codex(path: Path, calls: Path) -> Path:
    path.write_text(
        f"""#!{sys.executable}
import json, pathlib, re, sys
args=sys.argv[1:]
last=pathlib.Path(args[args.index('--output-last-message')+1])
prompt=sys.stdin.read()
scores=[int(value) for value in re.findall(r'SCORE = (\\d+)', prompt)]
score=max(scores)
calls=pathlib.Path({str(calls)!r})
number=sum(1 for _ in calls.open())+1 if calls.exists() else 1
resumed='resume' in args
session=args[-2] if resumed else f'semantic-session-{{number}}'
last.write_text(
    'MECHANISM: increment the executable score\\n'
    'HYPOTHESIS: a larger score improves the toy objective\\n'
    'INTENDED_EDIT: increment SCORE\\n'
    'EVIDENCE: prior increments were valid\\n'
    f'<<<<<<< SEARCH\\nSCORE = {{score}}\\n=======\\n'
    f'SCORE = {{score+1}}\\n>>>>>>> REPLACE\\n'
)
with calls.open('a') as handle:
    handle.write(json.dumps({{
        'number': number, 'resumed': resumed, 'session': session
    }})+'\\n')
print(json.dumps({{'type':'thread.started','thread_id':session}}))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':10,'cached_input_tokens':1,'output_tokens':2,'reasoning_output_tokens':1}}}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_semantic_campaign_shares_prefix_then_forks(tmp_path: Path) -> None:
    spec = _spec()
    task = _task(_seed(tmp_path / "seed"))
    framework = FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_direct_editor_v1",
        prompt_profile="controlled_factorial_v1",
        edit_mode="direct_workspace",
    )
    baseline = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task,
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_semantic_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task,
        framework=framework,
        intervention_plan_path=_plan(tmp_path / "interventions.toml"),
        calibration_path=baseline,
        repo_root=ROOT,
    )
    assert validate_semantic_campaign(campaign, repo_root=ROOT)["valid"]
    schedule = json.loads((campaign / "schedule.json").read_text())
    by_condition = {row["condition"]: row["run_id"] for row in schedule}
    set_semantic_run_control(
        campaign,
        run_id=by_condition["assumption_challenge"],
        desired="paused",
        reason="test independent arm pause",
    )
    status = semantic_status(campaign)
    desired = {row["run_id"]: row["desired"] for row in status["runs"]}
    assert desired[by_condition["assumption_challenge"]] == "paused"
    assert desired[by_condition["passive_control"]] == "running"
    set_semantic_run_control(
        campaign,
        run_id=by_condition["assumption_challenge"],
        desired="running",
        reason="test independent arm resume",
    )
    counter = tmp_path / "calls"
    codex = _codex(tmp_path / "fake-codex", counter)

    first = run_semantic_opportunity(
        campaign,
        run_id=by_condition["assumption_challenge"],
        repo_root=ROOT,
        python_bin=sys.executable,
        codex_binary=str(codex),
    )
    assert first["shared_prefix"] is True
    assert counter.read_text() == "1"
    states = [
        SearchController.load(campaign / "runs" / row["run_id"], spec).state
        for row in schedule
    ]
    assert {state.incumbent_id for state in states} == {states[0].incumbent_id}
    assert {state.proposals_used for state in states} == {1}

    for run_id in by_condition.values():
        run_semantic_opportunity(
            campaign,
            run_id=run_id,
            repo_root=ROOT,
            python_bin=sys.executable,
            codex_binary=str(codex),
        )
    assert counter.read_text() == "3"
    prompts = {
        condition: (
            campaign / "runs" / run_id / "opportunities/0002/prompt.md"
        ).read_text()
        for condition, run_id in by_condition.items()
    }
    assert "load-bearing assumption" in prompts["assumption_challenge"]
    assert "load-bearing assumption" not in prompts["passive_control"]


def test_openevolve_uses_five_proposal_sessions_and_resets_at_fork(
    tmp_path: Path,
) -> None:
    spec = _phased_spec()
    task = _task(_seed(tmp_path / "seed"))
    framework = FrameworkSpec(
        framework_id=FrameworkKind.OPENEVOLVE,
        adapter="controlled_openevolve_prompt_diff_v3",
        prompt_profile="fashion_mnist_openevolve_v2_1",
        edit_mode="search_replace_diff",
    )
    plan = tmp_path / "interventions.toml"
    plan.write_text(
        (_plan(tmp_path / "template-plan.toml"))
        .read_text(encoding="utf-8")
        .replace("shared_prefix_opportunities = 1", "shared_prefix_opportunities = 5")
        .replace("session_span_opportunities = 1", "session_span_opportunities = 5"),
        encoding="utf-8",
    )
    baseline = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task,
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_semantic_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task,
        framework=framework,
        intervention_plan_path=plan,
        calibration_path=baseline,
        repo_root=ROOT,
    )
    schedule = json.loads((campaign / "schedule.json").read_text())
    by_condition = {row["condition"]: row["run_id"] for row in schedule}
    calls = tmp_path / "calls.jsonl"
    codex = _openevolve_codex(tmp_path / "fake-codex", calls)

    for _ in range(5):
        result = run_semantic_opportunity(
            campaign,
            run_id=by_condition["assumption_challenge"],
            repo_root=ROOT,
            python_bin=sys.executable,
            codex_binary=str(codex),
        )
        assert result["shared_prefix"] is True
    run_semantic_opportunity(
        campaign,
        run_id=by_condition["passive_control"],
        repo_root=ROOT,
        python_bin=sys.executable,
        codex_binary=str(codex),
    )

    call_rows = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [row["resumed"] for row in call_rows] == [
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert {row["session"] for row in call_rows[:5]} == {"semantic-session-1"}
    assert call_rows[5]["session"] == "semantic-session-6"
    leader = campaign / "runs" / by_condition["passive_control"]
    reset_events = [
        json.loads(line)
        for line in (leader / "events.jsonl").read_text().splitlines()
        if '"conversation_session_reset"' in line
    ]
    assert len(reset_events) == 1
    assert reset_events[0]["opportunity"] == 6


def test_training_ladder_screens_and_escalates(tmp_path: Path) -> None:
    task = _task(_seed(tmp_path / "seed"))
    outputs = {
        25_000: 0.85,
        50_000: 0.86,
        100_000: 0.90,
        5_000: 0.80,
        10_000: 0.99,
    }

    class Evaluator:
        def __init__(self, selected: TaskSpec):
            command = list(selected.evaluator_command)
            flag = (
                "--training-examples"
                if "--training-examples" in command
                else "--max-steps"
            )
            self.level = int(command[command.index(flag) + 1])
            self.qualification_minimum = selected.qualification_minimum

        def evaluate(self, **_kwargs):
            accuracy = outputs[self.level]
            qualified = self.qualification_minimum is None or (
                accuracy >= self.qualification_minimum
            )
            return SimpleNamespace(
                evaluation=Evaluation(
                    valid=qualified,
                    fitness=accuracy if qualified else None,
                    metrics={
                        "score": accuracy,
                        "accuracy": accuracy,
                        "validation_accuracy": accuracy,
                    },
                    evaluator_seconds=1.0,
                    failure_kind=None if qualified else "nonqualification",
                )
            )

    screened = evaluate_training_ladder(
        task=task,
        config={
            "strategy": "successive_screen_then_full_confirmation_v1",
            "training_examples": [25_000, 50_000, 100_000],
            "promotion_validation_accuracy": [0.82, 0.87, None],
        },
        candidate_snapshot=tmp_path,
        opportunity_root=tmp_path / "screen",
        timeout_seconds=10,
        run_seed=1,
        evaluator_factory=Evaluator,
    )
    assert screened.valid is False
    assert screened.failure_kind == "fidelity_screen_not_promoted"
    assert screened.evaluator_seconds == 2.0

    (tmp_path / "train.py").write_text(
        "EVALUATION_LADDER = [25_000, 50_000]\n"
        "EVALUATION_PROMOTION_THRESHOLDS = [0.82, 0.85]\n",
        encoding="utf-8",
    )
    customized = evaluate_training_ladder(
        task=task,
        config={
            "strategy": "successive_screen_then_full_confirmation_v1",
            "training_examples": [25_000, 50_000, 100_000],
            "command_argument": "--training-examples",
            "promotion_validation_accuracy": [0.82, 0.87, None],
            "candidate_editable_policy": {
                "enabled": True,
                "path": "train.py",
                "levels_symbol": "EVALUATION_LADDER",
                "thresholds_symbol": "EVALUATION_PROMOTION_THRESHOLDS",
                "minimum_level": 10_000,
                "maximum_level": 100_000,
                "required_terminal_level": 100_000,
                "max_rungs": 6,
            },
        },
        candidate_snapshot=tmp_path,
        opportunity_root=tmp_path / "screen-custom",
        timeout_seconds=10,
        run_seed=1,
        evaluator_factory=Evaluator,
    )
    assert customized.valid is True
    custom_receipt = json.loads(
        (
            tmp_path
            / "screen-custom/fidelity/ladder-result.json"
        ).read_text(encoding="utf-8")
    )["candidate_editable_policy"]
    assert custom_receipt["accepted"] is True
    assert custom_receipt["levels"] == [25_000, 50_000, 100_000]

    qualified_task = _task(tmp_path / "seed", qualification=True)
    qualified_task = TaskSpec(
        **{
            **qualified_task.__dict__,
            "evaluator_command": (
                "{python}",
                "evaluate.py",
                "--max-steps",
                "5000",
                "--output",
                "{output}",
            ),
        }
    )
    escalated = evaluate_training_ladder(
        task=qualified_task,
        config={
            "strategy": "escalate_until_qualified_v1",
            "levels": [5_000, 10_000],
            "command_argument": "--max-steps",
        },
        candidate_snapshot=tmp_path,
        opportunity_root=tmp_path / "escalate",
        timeout_seconds=10,
        run_seed=1,
        evaluator_factory=Evaluator,
    )
    assert escalated.valid is True
    assert escalated.metrics["fidelity_qualification_level"] == 10_000
    assert len(escalated.metrics["fidelity_stages"]) == 2


def test_developmental_archive_preserves_executable_near_miss(tmp_path: Path) -> None:
    seed = _seed(tmp_path / "seed")
    task = _task(seed, qualification=True)
    run_dir = tmp_path / "run"
    controller = SearchController.create(
        run_dir,
        _spec(),
        run_id="near-miss",
        condition=Condition.C0,
        seed_candidate=Candidate(
            candidate_id="baseline",
            parent_ids=[],
            fitness=1.0,
            metrics={"score": 1.0, "accuracy": 0.99},
            artifact_path="candidates/baseline",
            hypothesis="baseline",
            intended_edit="baseline",
            created_opportunity=0,
            retained_order=0,
        ),
    )
    record = {
        "run_id": "near-miss",
        "opportunity": 1,
        "candidate_id": "candidate",
        "parent_ids": ["baseline"],
        "retained": False,
        "mechanism": "smaller alternate representation",
        "evaluation": {
            "valid": False,
            "fitness": None,
            "failure_kind": "training_ladder_exhausted_without_qualification",
            "metrics": {"score": 0.98, "accuracy": 0.98},
        },
    }
    (run_dir / "opportunities/0001").mkdir(parents=True)
    assessment = assess_developmental_value(
        run_dir=run_dir,
        task=task,
        record=record,
        provenance={"semantic_delta_fingerprint": "new-mechanism"},
        config={
            "enabled": True,
            "selection_effect": "none",
            "archive_capacity": 8,
            "near_qualification_absolute_margin": 0.02,
        },
    )
    assert controller.state.candidates["baseline"].fitness == 1.0
    assert assessment["status"] == "provisional_nonqualifying"
    assert assessment["execution_valid"] is True
    assert assessment["near_qualification"] is True
    archive = json.loads((run_dir / "developmental-archive.json").read_text())
    assert archive["items"][0]["candidate_id"] == "candidate"
    assert archive["selection_effect"] == "none"

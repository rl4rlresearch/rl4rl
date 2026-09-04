# ruff: noqa: E402 -- the repository root is inserted for standalone test runs.

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.c0c3_factorial.campaign import calibrate_task, create_campaign
from experiments.c0c3_factorial.orchestration import run_v3_paired_opportunity
from experiments.c0c3_factorial.spec import (
    UNIFIED_V3_EXECUTION_RULE,
    BudgetSpec,
    ConversationMode,
    ExecutionBackend,
    FactorialSpec,
    FrameworkKind,
    FrameworkSpec,
    ModelSpec,
    ObjectiveDirection,
    TaskSpec,
)
from experiments.c0c3_factorial.state import SearchController
from experiments.c0c3_factorial.v3 import snapshot_prompt_bundle, v3_health_report
from experiments.c0c3_factorial.v3_analysis import audit_campaign
from experiments.c0c3_factorial.validation import validate_campaign


def _spec(study_id: str) -> FactorialSpec:
    return FactorialSpec(
        protocol_version="3.0",
        study_id=study_id,
        study_seed=123,
        blocks=1,
        portfolio_capacity=2,
        transition_opportunities=(2,),
        conversation_mode=ConversationMode.BOUNDED,
        model=ModelSpec("gpt-fake", "high"),
        budget=BudgetSpec(
            proposals=3,
            candidate_evaluations=3,
            max_total_tokens=1000000,
            max_evaluator_seconds=100.0,
            evaluator_timeout_seconds=10,
        ),
        execution_rule=UNIFIED_V3_EXECUTION_RULE,
        include_no_search=False,
    )


def _seed(root: Path) -> Path:
    root.mkdir()
    (root / "candidate.py").write_text("SCORE = 0\n", encoding="utf-8")
    (root / "evaluate.py").write_text(
        """import argparse, json
from candidate import SCORE
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
open(a.output, 'w').write(json.dumps({'metrics': {'score': SCORE}}))
""",
        encoding="utf-8",
    )
    return root


def _task(seed: Path) -> TaskSpec:
    command = ("{python}", "evaluate.py", "--output", "{output}")
    return TaskSpec(
        task_id="v3-toy",
        display_name="Toy optimization",
        adapter="command_json_v1",
        seed_source=str(seed),
        editable_paths=("candidate.py",),
        evaluator_command=command,
        objective_metric="score",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        qualification_metric=None,
        qualification_minimum=None,
        public_feedback_metrics=("score",),
        metric_patterns={"score": r"score:\s*([0-9.]+)"},
        final_holdout_command=command,
        preferred_backend=ExecutionBackend.LOCAL,
    )


def _autoresearch_codex(path: Path, counter: Path) -> Path:
    path.write_text(
        f"""#!{sys.executable}
import json, pathlib, sys
args = sys.argv[1:]
workspace = pathlib.Path(args[args.index('--cd') + 1])
last = pathlib.Path(args[args.index('--output-last-message') + 1])
prompt = sys.stdin.read()
candidate = workspace / 'candidate.py'
score = int(candidate.read_text().split('=')[1])
candidate.write_text(f'SCORE = {{score + 1}}\\n')
last.write_text('HYPOTHESIS: improve score\\nINTENDED_EDIT: increment score\\n')
counter = pathlib.Path({str(counter)!r})
counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
print(json.dumps({{'type':'thread.started','thread_id':'ephemeral'}}))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':10,'cached_input_tokens':2,'output_tokens':3,'reasoning_output_tokens':1}}}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _openevolve_codex(path: Path, counter: Path) -> Path:
    path.write_text(
        f"""#!{sys.executable}
import json, pathlib, re, sys
args = sys.argv[1:]
last = pathlib.Path(args[args.index('--output-last-message') + 1])
prompt = sys.stdin.read()
match = re.search(r'SCORE = (\\d+)', prompt)
score = int(match.group(1))
last.write_text(
    f'<<<<<<< SEARCH\\nSCORE = {{score}}\\n=======\\n'
    f'SCORE = {{score + 1}}\\n>>>>>>> REPLACE\\n'
)
counter = pathlib.Path({str(counter)!r})
counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
print(json.dumps({{'type':'thread.started','thread_id':'ephemeral'}}))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':10,'cached_input_tokens':2,'output_tokens':3,'reasoning_output_tokens':1}}}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _campaign(
    tmp_path: Path, *, spec: FactorialSpec, framework: FrameworkSpec
) -> tuple[Path, TaskSpec]:
    task = _task(_seed(tmp_path / "seed"))
    calibration = tmp_path / "calibration"
    baseline = calibrate_task(
        calibration,
        spec=spec,
        task=task,
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task,
        framework=framework,
        calibration_path=baseline,
        repo_root=ROOT,
    )
    snapshot_prompt_bundle(
        campaign,
        spec=spec,
        framework=framework,
        repo_root=ROOT,
    )
    return campaign, task


def _run_id(campaign: Path, condition: str) -> str:
    schedule = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    return next(str(row["run_id"]) for row in schedule if row["condition"] == condition)


def test_v3_autoresearch_uses_one_physical_prefix_then_forks(tmp_path: Path) -> None:
    spec = _spec("v3-ar-test")
    framework = FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_direct_editor_v1",
        prompt_profile="controlled_factorial_v1",
        edit_mode="direct_workspace",
    )
    campaign, task = _campaign(tmp_path, spec=spec, framework=framework)
    assert validate_campaign(
        campaign,
        spec=spec,
        task=task,
        framework=framework,
        repo_root=ROOT,
    )["valid"]
    counter = tmp_path / "calls"
    codex = _autoresearch_codex(tmp_path / "fake-codex", counter)
    c0 = _run_id(campaign, "C0")
    c1 = _run_id(campaign, "C1")

    shared = run_v3_paired_opportunity(
        campaign,
        spec=spec,
        task=task,
        framework=framework,
        repo_root=ROOT,
        python_bin=sys.executable,
        run_id=c1,
        codex_binary=str(codex),
    )
    assert shared["shared_prefix"] is True
    assert counter.read_text() == "1"
    c0_state = SearchController.load(campaign / "runs" / c0, spec).state
    c1_state = SearchController.load(campaign / "runs" / c1, spec).state
    assert c0_state.incumbent_id == c1_state.incumbent_id
    assert c0_state.proposals_used == c1_state.proposals_used == 1

    for run_id in (c0, c1):
        run_v3_paired_opportunity(
            campaign,
            spec=spec,
            task=task,
            framework=framework,
            repo_root=ROOT,
            python_bin=sys.executable,
            run_id=run_id,
            codex_binary=str(codex),
        )
    assert counter.read_text() == "3"
    assert (
        json.loads(
            (campaign / "runs" / c1 / "opportunities/0002/result.json").read_text()
        )["proposal_type"]
        == "assumption_changing"
    )
    assert v3_health_report(campaign, spec=spec, task=task, framework=framework)[
        "valid"
    ]
    assert audit_campaign(campaign, spec=spec, task=task)["valid"]


def test_v3_openevolve_prefix_accepts_patch_without_prose_metadata(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("dacite") is None:
        import pytest

        pytest.skip("OpenEvolve dependencies are installed in the architecture runtime")
    spec = _spec("v3-oe-test")
    framework = FrameworkSpec(
        framework_id=FrameworkKind.OPENEVOLVE,
        adapter="controlled_openevolve_prompt_diff_v3",
        prompt_profile="controlled_factorial_v1",
        edit_mode="search_replace_diff",
    )
    campaign, task = _campaign(tmp_path, spec=spec, framework=framework)
    counter = tmp_path / "oe-calls"
    codex = _openevolve_codex(tmp_path / "fake-oe-codex", counter)
    c2 = _run_id(campaign, "C2")
    c3 = _run_id(campaign, "C3")

    record = run_v3_paired_opportunity(
        campaign,
        spec=spec,
        task=task,
        framework=framework,
        repo_root=ROOT,
        python_bin=sys.executable,
        run_id=c3,
        codex_binary=str(codex),
    )
    assert record["shared_prefix"] is True
    assert record["evaluation"]["valid"] is True
    assert counter.read_text() == "1"
    left = SearchController.load(campaign / "runs" / c2, spec).state
    right = SearchController.load(campaign / "runs" / c3, spec).state
    assert left.portfolio_ids == right.portfolio_ids
    assert left.proposals_used == right.proposals_used == 1

# ruff: noqa: E402 -- the standalone experiments package is added explicitly.

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.c0c3_factorial.artifacts import (
    candidate_hash,
    materialize_candidate,
    snapshot_candidate,
)
from experiments.c0c3_factorial.campaign import calibrate_task, create_campaign
from experiments.c0c3_factorial.codex_cli import CodexCli
from experiments.c0c3_factorial.frameworks import (
    OpenEvolveAdapter,
    bundle_workspace,
    parse_metadata,
    unbundle_workspace,
)
from experiments.c0c3_factorial.prompts import RenderedPrompt
from experiments.c0c3_factorial.runner import run_one_opportunity
from experiments.c0c3_factorial.spec import (
    BudgetSpec,
    ExecutionBackend,
    FactorialSpec,
    FrameworkKind,
    FrameworkSpec,
    ModelSpec,
    ObjectiveDirection,
    TaskSpec,
)
from experiments.c0c3_factorial.state import SearchController


def protocol() -> FactorialSpec:
    return FactorialSpec(
        protocol_version="1.0",
        study_id="execution-test",
        study_seed=42,
        blocks=1,
        portfolio_capacity=2,
        transition_opportunities=(1,),
        model=ModelSpec("gpt-fake", "high"),
        budget=BudgetSpec(
            proposals=1,
            candidate_evaluations=1,
            max_total_tokens=1000,
            max_evaluator_seconds=100.0,
            evaluator_timeout_seconds=10,
        ),
    )


def task(seed_source: Path) -> TaskSpec:
    return TaskSpec(
        task_id="toy_execution",
        display_name="Toy execution task",
        adapter="command_json_v1",
        seed_source=str(seed_source),
        editable_paths=("candidate.py",),
        evaluator_command=(
            "{python}",
            "evaluate.py",
            "--output",
            "{output}",
        ),
        objective_metric="score",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        qualification_metric=None,
        qualification_minimum=None,
        public_feedback_metrics=("score",),
        metric_patterns={"score": r"score:\s*([0-9.]+)"},
        final_holdout_command=("{python}", "evaluate.py"),
        preferred_backend=ExecutionBackend.LOCAL,
    )


def framework() -> FrameworkSpec:
    return FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_direct_editor_v1",
        prompt_profile="controlled_factorial_v1",
        diff_mode=True,
    )


def make_seed(root: Path) -> Path:
    root.mkdir()
    (root / "candidate.py").write_text("SCORE = 0\n", encoding="utf-8")
    (root / "evaluate.py").write_text(
        """\
import argparse
import json
from candidate import SCORE

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
with open(args.output, 'w') as handle:
    json.dump({'metrics': {'score': SCORE}}, handle)
print(f'score: {SCORE}')
""",
        encoding="utf-8",
    )
    return root


def make_fake_codex(path: Path) -> Path:
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys

args = sys.argv[1:]
workspace = pathlib.Path(args[args.index('--cd') + 1])
last = pathlib.Path(args[args.index('--output-last-message') + 1])
_prompt = sys.stdin.read()
(workspace / 'candidate.py').write_text('SCORE = 1\\n')
last.write_text(
    'HYPOTHESIS: increasing the toy score improves fitness\\n'
    'INTENDED_EDIT: set SCORE to 1\\n'
)
print(json.dumps({{'type': 'thread.started', 'thread_id': 'fake'}}))
print(json.dumps({{
    'type': 'turn.completed',
    'usage': {{
        'input_tokens': 11,
        'cached_input_tokens': 3,
        'output_tokens': 5,
        'reasoning_output_tokens': 2,
    }},
}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_fake_diff_codex(path: Path) -> Path:
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys

args = sys.argv[1:]
last = pathlib.Path(args[args.index('--output-last-message') + 1])
_prompt = sys.stdin.read()
last.write_text(
    'HYPOTHESIS: a larger score improves the toy objective\\n'
    'INTENDED_EDIT: set SCORE to 1\\n'
    '<<<<<<< SEARCH\\nSCORE = 0\\n=======\\nSCORE = 1\\n>>>>>>> REPLACE\\n'
)
print(json.dumps({{'type': 'turn.completed', 'usage': {{
    'input_tokens': 8, 'cached_input_tokens': 2, 'output_tokens': 4,
    'reasoning_output_tokens': 1
}}}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_candidate_snapshot_and_bundle_round_trip(tmp_path: Path) -> None:
    workspace = make_seed(tmp_path / "seed")
    identifier, snapshot = snapshot_candidate(
        workspace, tmp_path / "candidates", ("candidate.py",)
    )
    assert identifier == candidate_hash(workspace, ("candidate.py",))
    materialized = tmp_path / "materialized"
    materialize_candidate(workspace, snapshot, materialized, ("candidate.py",))
    bundle = bundle_workspace(materialized, ("candidate.py",))
    (materialized / "candidate.py").write_text("broken", encoding="utf-8")
    unbundle_workspace(bundle, materialized, ("candidate.py",))
    assert (materialized / "candidate.py").read_text() == "SCORE = 0\n"


def test_metadata_parser() -> None:
    hypothesis, edit = parse_metadata(
        "HYPOTHESIS: use a different family\nINTENDED_EDIT: replace attention\n"
    )
    assert hypothesis == "use a different family"
    assert edit == "replace attention"


def test_codex_transport_campaign_and_one_real_controller_step(
    tmp_path: Path,
) -> None:
    seed_source = make_seed(tmp_path / "source")
    fake_codex = make_fake_codex(tmp_path / "fake-codex")
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=protocol(),
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=protocol(),
        task=task(seed_source),
        framework=framework(),
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    schedule = json.loads((campaign / "schedule.json").read_text())
    c0 = next(row for row in schedule if row["condition"] == "C0")
    record = run_one_opportunity(
        campaign / "runs" / c0["run_id"],
        spec=protocol(),
        task=task(seed_source),
        framework=framework(),
        repo_root=ROOT,
        python_bin=sys.executable,
        codex_binary=str(fake_codex),
        codex_timeout_seconds=10,
    )
    assert record["retained"] is True
    assert record["evaluation"]["metrics"]["score"] == 1
    assert record["usage_increment"]["total_tokens"] == 16
    controller = SearchController.load(campaign / "runs" / c0["run_id"], protocol())
    assert controller.state.status == "completed"
    assert controller.state.proposals_used == 1
    assert controller.state.evaluations_used == 1
    n0 = next(row for row in schedule if row["condition"] == "N0")
    n0_state = json.loads((campaign / "runs" / n0["run_id"] / "state.json").read_text())
    assert n0_state["no_search"] is True
    assert n0_state["condition"] == "N0"


def test_codex_event_usage_parser_uses_final_turn(tmp_path: Path) -> None:
    fake_codex = make_fake_codex(tmp_path / "fake-codex")
    workspace = make_seed(tmp_path / "workspace")
    result = CodexCli(str(fake_codex)).run(
        prompt="test",
        workspace=workspace,
        model=ModelSpec("gpt-fake", "high"),
        log_root=tmp_path / "logs",
        call_id="one",
    )
    assert result.returncode == 0
    assert result.usage.input_tokens == 11
    assert result.usage.cached_input_tokens == 3
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 16


def test_controlled_openevolve_uses_vendor_prompt_and_diff_primitives(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("dacite") is None:
        import pytest

        pytest.skip("OpenEvolve dependencies live in architecture_discovery/.venv")
    workspace = make_seed(tmp_path / "workspace")
    visible = make_seed(tmp_path / "visible")
    adapter = OpenEvolveAdapter(
        CodexCli(str(make_fake_diff_codex(tmp_path / "fake-diff-codex"))),
        vendor_root=ROOT / "architecture_discovery/vendor/openevolve",
    )
    rendered = RenderedPrompt(
        text="controlled system prompt",
        common_template_sha256="a",
        search_state_sha256="b",
        proposal_policy_sha256="c",
        prompt_sha256="d",
        transition_active=False,
    )
    result = adapter.propose(
        rendered=rendered,
        workspace=workspace,
        model=ModelSpec("gpt-fake", "high"),
        log_root=tmp_path / "logs",
        call_id="proposal-1",
        timeout_seconds=10,
        task=task(workspace),
        visible_workspaces=(visible,),
        selected_parent_id="seed",
        visible_records=({"candidate_id": "seed", "metrics": {"score": 0.0}},),
    )
    assert result.adapter_error is None
    assert (workspace / "candidate.py").read_text() == "SCORE = 1\n"

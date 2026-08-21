# ruff: noqa: E402 -- the standalone experiments package is added explicitly.

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.c0c3_factorial.artifacts import (
    candidate_hash,
    materialize_candidate,
    snapshot_candidate,
)
from experiments.c0c3_factorial.campaign import (
    calibrate_task,
    create_campaign,
    execute_calibration,
    prepare_calibration,
)
from experiments.c0c3_factorial.codex_cli import CodexCli
from experiments.c0c3_factorial.evaluator import CommandEvaluator
from experiments.c0c3_factorial.frameworks import (
    OpenEvolveAdapter,
    bundle_workspace,
    parse_metadata,
    unbundle_workspace,
)
from experiments.c0c3_factorial.modal_app import safe_campaign_path
from experiments.c0c3_factorial.orchestration import (
    CampaignLockedError,
    ParallelWaveError,
    campaign_lock,
    next_parallel_wave,
    next_run,
    run_parallel_campaign,
)
from experiments.c0c3_factorial.postsearch import (
    export_layer_b_packets,
    run_layer_c,
    score_layer_b,
)
from experiments.c0c3_factorial.prompts import RenderedPrompt
from experiments.c0c3_factorial.runner import (
    recover_active_opportunity,
    run_one_opportunity,
)
from experiments.c0c3_factorial.spec import (
    PARALLEL_EXECUTION_RULE,
    BudgetSpec,
    ExecutionBackend,
    FactorialSpec,
    FrameworkKind,
    FrameworkSpec,
    ModelSpec,
    ObjectiveDirection,
    TaskSpec,
)
from experiments.c0c3_factorial.state import Evaluation, SearchController, Usage
from experiments.c0c3_factorial.validation import validate_campaign


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


def parallel_protocol() -> FactorialSpec:
    serial = protocol()
    return FactorialSpec(
        **{
            **serial.__dict__,
            "protocol_version": "1.1",
            "study_id": "parallel-execution-test",
            "execution_rule": PARALLEL_EXECUTION_RULE,
        }
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
        final_holdout_command=(
            "{python}",
            "evaluate.py",
            "--output",
            "{output}",
        ),
        preferred_backend=ExecutionBackend.LOCAL,
    )


def framework() -> FrameworkSpec:
    return FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_direct_editor_v1",
        prompt_profile="controlled_factorial_v1",
        edit_mode="direct_workspace",
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
assert '-a' not in args
assert 'approval_policy="never"' in args
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


def make_parallel_barrier_fake_codex(path: Path, marker_root: Path) -> Path:
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys
import time

args = sys.argv[1:]
workspace = pathlib.Path(args[args.index('--cd') + 1])
last = pathlib.Path(args[args.index('--output-last-message') + 1])
run_id = workspace.parents[2].name
_prompt = sys.stdin.read()
if not run_id.endswith('-n0'):
    markers = pathlib.Path({str(marker_root)!r})
    markers.mkdir(parents=True, exist_ok=True)
    (markers / run_id).write_text('started')
    deadline = time.monotonic() + 5
    while len(list(markers.iterdir())) < 4:
        if time.monotonic() >= deadline:
            raise SystemExit('four factorial calls did not overlap')
        time.sleep(0.01)
(workspace / 'candidate.py').write_text('SCORE = 1\\n')
last.write_text(
    'HYPOTHESIS: increasing the toy score improves fitness\\n'
    'INTENDED_EDIT: set SCORE to 1\\n'
)
print(json.dumps({{'type': 'thread.started', 'thread_id': run_id}}))
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


def make_failing_fake_codex(path: Path) -> Path:
    path.write_text(
        f"""#!{sys.executable}
import sys

print('simulated provider transport failure', file=sys.stderr)
raise SystemExit(2)
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


def test_modal_campaign_path_is_relative_and_confined() -> None:
    assert str(safe_campaign_path("paper/adderboard")) == (
        "/campaigns/paper/adderboard"
    )
    for unsafe in ("/absolute", "../escape", "a/../../escape", "."):
        try:
            safe_campaign_path(unsafe)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"unsafe Modal path accepted: {unsafe}")


def test_campaign_writer_lock_rejects_a_second_owner(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    with (
        campaign_lock(campaign),
        pytest.raises(CampaignLockedError, match="active writer"),
        campaign_lock(campaign),
    ):
        pass


def test_evaluator_preserves_symlinked_virtualenv_interpreter(
    tmp_path: Path,
) -> None:
    seed_source = make_seed(tmp_path / "source")
    environment_bin = tmp_path / "environment" / "bin"
    environment_bin.mkdir(parents=True)
    environment_python = environment_bin / "python"
    environment_python.symlink_to(Path(sys.executable).resolve())

    evaluator = CommandEvaluator(
        task=task(seed_source),
        support_source=seed_source,
        repo_root=ROOT,
        python_bin=str(environment_python),
    )

    assert evaluator.python_bin == str(environment_python.absolute())
    assert Path(evaluator.python_bin).is_symlink()
    assert Path(evaluator.python_bin).resolve() == Path(sys.executable).resolve()


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
    report = validate_campaign(
        campaign,
        spec=protocol(),
        task=task(seed_source),
        framework=framework(),
        repo_root=ROOT,
    )
    assert report["valid"] is True
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


def test_portable_calibration_can_execute_on_a_later_backend(tmp_path: Path) -> None:
    seed_source = make_seed(tmp_path / "source")
    calibration = prepare_calibration(
        tmp_path / "calibration",
        spec=protocol(),
        task=task(seed_source),
        repo_root=ROOT,
    )
    assert not (calibration / "baseline.json").exists()
    baseline = execute_calibration(
        calibration,
        spec=protocol(),
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    assert json.loads(baseline.read_text())["calibration_kind"] == (
        "executed_on_target_backend"
    )


def test_runtime_hash_change_fails_validation_and_execution(tmp_path: Path) -> None:
    seed_source = make_seed(tmp_path / "source")
    baseline = calibrate_task(
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
        calibration_path=baseline,
        repo_root=ROOT,
        include_no_search=False,
    )
    assignment = json.loads((campaign / "schedule.json").read_text())[0]
    manifest_path = campaign / "runs" / assignment["run_id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scientific_runtime_hash"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_campaign(
        campaign,
        spec=protocol(),
        task=task(seed_source),
        framework=framework(),
        repo_root=ROOT,
    )
    assert report["valid"] is False
    assert any(
        "scientific_runtime_hash mismatch" in error for error in report["errors"]
    )
    try:
        run_one_opportunity(
            campaign / "runs" / assignment["run_id"],
            spec=protocol(),
            task=task(seed_source),
            framework=framework(),
            repo_root=ROOT,
            python_bin=sys.executable,
            codex_binary="not-used",
        )
    except ValueError as error:
        assert "scientific runtime changed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("execution accepted a mismatched runtime")


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
        run_seed=42,
    )
    assert result.adapter_error is None
    assert (workspace / "candidate.py").read_text() == "SCORE = 1\n"


def test_layer_b_is_sealed_until_completion_then_scores_factorial(
    tmp_path: Path,
) -> None:
    seed_source = make_seed(tmp_path / "source")
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
    try:
        export_layer_b_packets(campaign, spec=protocol(), task=task(seed_source))
    except RuntimeError as error:
        assert "sealed until every run completes" in str(error)
    else:  # pragma: no cover - fail explicitly if the seal regresses
        raise AssertionError("Layer B opened before search completed")
    schedule = json.loads((campaign / "schedule.json").read_text())
    for index, assignment in enumerate(schedule, start=1):
        run_dir = campaign / "runs" / assignment["run_id"]
        controller = SearchController.load(run_dir, protocol())
        controller.begin()
        seed_id = controller.state.incumbent_id
        shutil.copytree(
            run_dir / "candidates" / seed_id,
            run_dir / "candidates" / f"candidate-{index}",
        )
        controller.complete(
            candidate_id=f"candidate-{index}",
            artifact_path=f"candidates/{seed_id}",
            hypothesis=f"mechanism {index}",
            intended_edit="test one mechanism",
            evaluation=Evaluation(True, float(index), {"score": float(index)}, 0.1),
            usage=Usage(input_tokens=1, output_tokens=1),
            prompt_hashes={"prompt_sha256": f"hash-{index}"},
        )
    sealed = export_layer_b_packets(campaign, spec=protocol(), task=task(seed_source))
    packet_ids = [
        row["packet_id"]
        for row in csv.DictReader(
            (sealed / "packet_order.tsv").open(encoding="utf-8"), delimiter="\t"
        )
    ]
    first_packet = sealed / "packets" / packet_ids[0]
    assert (first_packet / "parent/candidate.py").is_file()
    packet_payload = json.loads((first_packet / "packet.json").read_text())
    assert "layer_a_metrics" not in packet_payload
    assert packet_payload["layer_a_qualified"] is True
    annotations = sealed / "annotations.tsv"
    with annotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "packet_id",
                "layer_b_qualified",
                "mechanism_cluster",
                "primary_mechanism",
                "reviewer",
                "notes",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        for packet_id in packet_ids:
            writer.writerow(
                {
                    "packet_id": packet_id,
                    "layer_b_qualified": "1",
                    "mechanism_cluster": "cluster-a",
                    "primary_mechanism": "toy",
                    "reviewer": "test",
                    "notes": "",
                }
            )
    scored = score_layer_b(campaign, annotations_path=annotations)
    estimates = json.loads((scored / "factorial_estimates.json").read_text())
    assert estimates["cell_means"] == {"C0": 1.0, "C1": 1.0, "C2": 1.0, "C3": 1.0}
    assert estimates["portfolio_memory_main_effect"] == 0.0
    layer_c = run_layer_c(
        campaign,
        spec=protocol(),
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    summaries = json.loads((layer_c / "summary.json").read_text())
    assert len(summaries) == 5
    assert all(row["layer_c_evaluation"]["valid"] for row in summaries)
    no_search = next(row for row in summaries if row["condition"] == "N0")
    assert no_search["selected_by_layer_a_candidate_id"] == "candidate-5"
    assert no_search["selection_rule"] == (
        "post_search_best_independent_layer_a_candidate"
    )


def test_campaign_orchestration_is_blocked_round_robin(tmp_path: Path) -> None:
    seed_source = make_seed(tmp_path / "source")
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
    first = next_run(campaign, protocol())
    assert first is not None
    assert first.run_id == schedule[0]["run_id"]
    first_controller = SearchController.load(
        campaign / "runs" / first.run_id, protocol()
    )
    first_controller.begin()
    first_controller.complete(
        candidate_id="finished-first",
        artifact_path=f"candidates/{first_controller.state.incumbent_id}",
        hypothesis="test ordering",
        intended_edit="none",
        evaluation=Evaluation(False, None, {}, 0.0, evaluator_calls=0),
        usage=Usage(input_tokens=1, output_tokens=1),
        prompt_hashes={},
    )
    second = next_run(campaign, protocol())
    assert second is not None
    assert second.run_id == schedule[1]["run_id"]


def test_parallel_wave_selects_only_lagging_factorial_cells(
    tmp_path: Path,
) -> None:
    spec = parallel_protocol()
    seed_source = make_seed(tmp_path / "source")
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task(seed_source),
        framework=framework(),
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    schedule = json.loads((campaign / "schedule.json").read_text())
    first = next(row for row in schedule if row["condition"] != "N0")
    controller = SearchController.load(campaign / "runs" / first["run_id"], spec)
    controller.begin()
    controller.complete(
        candidate_id="interrupted-peer",
        artifact_path=f"candidates/{controller.state.incumbent_id}",
        hypothesis="simulate a peer completing before host interruption",
        intended_edit="none",
        evaluation=Evaluation(False, None, {}, 0.0, evaluator_calls=0),
        usage=Usage(input_tokens=1, output_tokens=1),
        prompt_hashes={},
    )
    wave = next_parallel_wave(campaign, spec)
    assert wave is not None
    assert wave.recovery_subset is True
    assert first["run_id"] not in {run.run_id for run in wave.factorial_runs}
    assert len(wave.factorial_runs) == 3
    assert wave.no_search_run is not None


def test_parallel_campaign_overlaps_c0_c3_and_serializes_n0(
    tmp_path: Path,
) -> None:
    spec = parallel_protocol()
    seed_source = make_seed(tmp_path / "source")
    fake_codex = make_parallel_barrier_fake_codex(
        tmp_path / "fake-parallel-codex", tmp_path / "parallel-markers"
    )
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task(seed_source),
        framework=framework(),
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    report = validate_campaign(
        campaign,
        spec=spec,
        task=task(seed_source),
        framework=framework(),
        repo_root=ROOT,
    )
    assert report["valid"] is True
    assert report["controls"]["frozen_parallel_condition_rounds"] is True
    with pytest.raises(ValueError, match="serial campaign commands require"):
        next_run(campaign, spec)

    results = list(
        run_parallel_campaign(
            campaign,
            spec=spec,
            task=task(seed_source),
            framework=framework(),
            repo_root=ROOT,
            python_bin=sys.executable,
            codex_binary=str(fake_codex),
            codex_timeout_seconds=10,
            max_block_rounds=1,
        )
    )
    assert len(results) == 1
    assert {row["condition"] for row in results[0]["factorial_records"]} == {
        "C0",
        "C1",
        "C2",
        "C3",
    }
    assert results[0]["no_search_record"]["condition"] == "N0"
    assert len(list((tmp_path / "parallel-markers").iterdir())) == 4
    schedule = json.loads((campaign / "schedule.json").read_text())
    assert all(
        SearchController.load(campaign / "runs" / row["run_id"], spec).state.status
        == "completed"
        for row in schedule
    )
    events = [
        json.loads(line)
        for line in (campaign / "parallel-rounds.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in events] == [
        "parallel_wave_started",
        "parallel_wave_completed",
    ]


def test_parallel_campaign_stops_after_zero_token_provider_failure(
    tmp_path: Path,
) -> None:
    spec = parallel_protocol()
    seed_source = make_seed(tmp_path / "source")
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task(seed_source),
        framework=framework(),
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    failing_codex = make_failing_fake_codex(tmp_path / "failing-codex")

    with pytest.raises(ParallelWaveError, match="before token accounting"):
        list(
            run_parallel_campaign(
                campaign,
                spec=spec,
                task=task(seed_source),
                framework=framework(),
                repo_root=ROOT,
                python_bin=sys.executable,
                codex_binary=str(failing_codex),
                codex_timeout_seconds=10,
            )
        )

    schedule = json.loads((campaign / "schedule.json").read_text())
    states = {
        row["condition"]: SearchController.load(
            campaign / "runs" / row["run_id"], spec
        ).state
        for row in schedule
    }
    assert all(
        states[name].proposals_used == 1 for name in ("C0", "C1", "C2", "C3")
    )
    assert states["N0"].proposals_used == 0
    events = [
        json.loads(line)
        for line in (campaign / "parallel-rounds.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in events] == [
        "parallel_wave_started",
        "parallel_wave_failed",
    ]


def test_interrupted_opportunity_recovery_consumes_proposal_not_evaluation(
    tmp_path: Path,
) -> None:
    seed_source = make_seed(tmp_path / "source")
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
        include_no_search=False,
    )
    assignment = json.loads((campaign / "schedule.json").read_text())[0]
    run_dir = campaign / "runs" / assignment["run_id"]
    SearchController.load(run_dir, protocol()).begin()
    record = recover_active_opportunity(
        run_dir, spec=protocol(), reason="simulated host interruption"
    )
    assert record["evaluation"]["failure_kind"] == "infrastructure_interruption"
    assert record["evaluator_calls_increment"] == 0
    controller = SearchController.load(run_dir, protocol())
    assert controller.state.proposals_used == 1
    assert controller.state.evaluations_used == 0
    assert controller.state.status == "completed"

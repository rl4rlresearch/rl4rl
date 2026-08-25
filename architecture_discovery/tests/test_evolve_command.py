from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

import modal_app
from common.evolution_run import (
    EVOLUTION_ACTION,
    EVOLUTION_MAX_ITERATIONS,
    EvolutionRunSpec,
)
from common.runtime_context import ExecutionContextV1
from modal_boundary import build_modal_cli_command
from research_dynamics.contracts import (
    ConditionId,
    FrameworkKind,
    ProcessCondition,
    ProcessStudyConfig,
    build_modal_process_payload,
)
from scripts import launch_modal

ROOT = Path(__file__).resolve().parents[1]


def _context() -> ExecutionContextV1:
    return ExecutionContextV1(
        execution_backend="modal",
        run_id="evolution-run-1",
        app_name="rl4rl-architecture-discovery",
        function_name="evolution_run",
        modal_app_id="ap-evolve",
        modal_function_id="fu-evolve",
        modal_call_id="fc-evolve",
        modal_image_id="im-evolve",
        image_source_sha256="a" * 64,
        artifact_uri=(
            "volume://rl4rl-architecture-artifacts/runs/evolution-run-1"
        ),
    )


def test_evolution_spec_binds_iterations_and_dynamic_deadlines() -> None:
    spec = EvolutionRunSpec.from_cli("openevolve", 20)
    assert spec.token == "openevolve_generic-n20"
    assert spec.controller_timeout_seconds == 41_580
    assert spec.function_timeout_seconds == 41_880
    assert spec.outer_cli_timeout_seconds == 42_780
    assert EvolutionRunSpec.parse(spec.token) == spec
    treated = spec.with_process_payload("YWJj")
    assert EvolutionRunSpec.parse(treated.token) == treated
    assert treated.base_token == spec.token

    maximum = EvolutionRunSpec.from_cli(
        "semantic-openevolve", EVOLUTION_MAX_ITERATIONS
    )
    assert maximum.function_timeout_seconds <= 24 * 60 * 60
    with pytest.raises(ValueError, match="between 1 and"):
        EvolutionRunSpec.from_cli("openevolve", EVOLUTION_MAX_ITERATIONS + 1)


def test_modal_contract_binds_generic_evolution_spec() -> None:
    spec = EvolutionRunSpec.from_cli("autoresearch", 7)
    kwargs = {
        "python_executable": ROOT / ".venv" / "bin" / "python",
        "project_root": ROOT,
        "action": EVOLUTION_ACTION,
        "run_id": "evolution-run-1",
        "harness": spec.token,
        "source_tree_sha256": "a" * 64,
        "cohort_id": "evolution-cohort-1",
        "image_source_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="provider approval differs"):
        build_modal_cli_command(**kwargs, provider_approved=False)
    command = build_modal_cli_command(**kwargs, provider_approved=True)
    assert command[command.index("--action") + 1] == EVOLUTION_ACTION
    assert command[command.index("--harness") + 1] == spec.token
    assert launch_modal.expected_outer_cli_timeout_seconds(
        EVOLUTION_ACTION, spec.token
    ) == spec.outer_cli_timeout_seconds
    profile = launch_modal.modal_resource_profile(EVOLUTION_ACTION, spec.token)
    assert profile["runtime_function_calls"][0]["function_name"] == "evolution_run"
    assert profile["runtime_function_calls"][0]["timeout_seconds"] == (
        spec.function_timeout_seconds
    )


def test_modal_action_constructs_requested_controller_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run_command(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return {"returncode": 0}

    monkeypatch.setattr(modal_app, "_run_command", fake_run_command)
    spec = EvolutionRunSpec.from_cli("semantic-autoresearch", 9)
    result = modal_app._evolution_action(tmp_path, _context(), spec=spec)
    command = captured["command"]
    assert command[command.index("--iterations") + 1] == "9"
    assert command[command.index("--seed") + 1] == "1"
    assert command[command.index("--training-profile") + 1] == (
        "trajectory_train_cuda_v2"
    )
    assert "--modal-evolution-run" in command
    assert captured["kwargs"]["timeout_seconds"] == (
        spec.controller_timeout_seconds
    )
    assert captured["kwargs"]["function_timeout_seconds"] == (
        spec.function_timeout_seconds
    )
    assert result["evolution_spec"] == spec.token


def test_modal_action_materializes_process_intervention(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run_command(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return {"returncode": 0}

    checkpoint = tmp_path / "checkpoint.ir.json"
    checkpoint.write_text('{"schema_version":"1.0"}\n', encoding="utf-8")
    config = ProcessStudyConfig(
        study_id="modal-process-test",
        run_id="modal-process-test-rd3",
        framework=FrameworkKind.OPENEVOLVE,
        condition=ProcessCondition.for_id(ConditionId.RD3),
        challenge_opportunities=(1,),
        source_checkpoint_id="checkpoint-test",
        source_checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    payload = build_modal_process_payload(config, initial_candidate=checkpoint)
    spec = EvolutionRunSpec.from_cli("openevolve", 2).with_process_payload(payload)
    monkeypatch.setattr(modal_app, "_run_command", fake_run_command)

    result = modal_app._evolution_action(tmp_path, _context(), spec=spec)

    environment = captured["kwargs"]["extra_environment"]
    config_path = Path(environment["RL4RL_PROCESS_CONFIG"])
    candidate_path = Path(environment["RL4RL_PROCESS_INITIAL_CANDIDATE"])
    assert config_path.parent == tmp_path / "process_inputs"
    assert ProcessStudyConfig.from_dict(
        json.loads(config_path.read_text(encoding="utf-8"))
    ) == config
    assert candidate_path.read_bytes() == checkpoint.read_bytes()
    assert result["evolution_spec"] == spec.base_token
    assert result["research_process"]["config_hash"] == config.config_hash
    assert result["research_process"]["condition_id"] == "RD3"


def test_top_level_command_plans_without_starting_paid_work() -> None:
    completed = subprocess.run(
        [str(ROOT / "evolve"), "openevolve", "-n", "5"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    payload = json.loads(completed.stdout)
    assert payload["evolution_spec"] == "openevolve_generic-n5"
    assert payload["provider_request_ceiling"] == 5
    assert payload["training_profile"] == "trajectory_train_cuda_v2"
    assert payload["optimizer_steps_per_candidate"] == 5_000
    assert payload["global_batch_size"] == 512
    assert payload["training_examples_per_candidate"] == 2_560_000
    assert payload["paid_work_started"] is False


def test_top_level_command_attaches_environment_process_config(
    tmp_path: Path,
) -> None:
    config = ProcessStudyConfig(
        study_id="modal-plan-test",
        run_id="modal-plan-test-rd0",
        framework=FrameworkKind.AUTORESEARCH,
        condition=ProcessCondition.for_id(ConditionId.RD0),
        challenge_opportunities=(),
    )
    config_path = tmp_path / "process_config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["RL4RL_PROCESS_CONFIG"] = str(config_path)
    completed = subprocess.run(
        [str(ROOT / "evolve"), "autoresearch", "-n", "2"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    payload = json.loads(completed.stdout)
    assert payload["evolution_spec"] == "greedy_autoresearch-n2"
    assert payload["process_intervention_attached"] is True
    assert len(payload["process_transport_sha256"]) == 64


def test_top_level_command_recognizes_estimated_cost_approval() -> None:
    completed = subprocess.run(
        [
            str(ROOT / "evolve"),
            "openevolve",
            "-n",
            "5",
            "--accept-estimated-cost",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode != 0
    assert "--accept-estimated-cost requires --execute" in completed.stderr
    assert "unrecognized arguments" not in completed.stderr

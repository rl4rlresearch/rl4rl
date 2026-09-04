#!/usr/bin/env python3
"""Controller for an audited C0-C3 block extension.

The original controller package remains frozen in the detached runtime.  This
operational wrapper supplies the same per-trajectory lifecycle, prompt
snapshot, locking, pause, and recovery semantics for schedule blocks appended
by ``c0c3_campaign_amend.py``; it deliberately does not alter any pre-existing
trajectory or its runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_LOCAL_EVALUATOR_CAPACITY_ENV = "RL4RL_SHARED_LOCAL_EVALUATOR_CAPACITY"
CAMPAIGN_LOCAL_EVALUATOR_CAPACITY_ENV = "RL4RL_CAMPAIGN_LOCAL_EVALUATOR_CAPACITY"


def _runtime_root(value: str | None) -> Path:
    root = Path(value or os.getcwd()).expanduser().resolve()
    if not (root / "experiments/c0c3_factorial").is_dir():
        raise ValueError(f"runtime root has no C0-C3 controller package: {root}")
    return root


def _load_runtime(root: Path) -> dict[str, Any]:
    """Import the frozen controller only after selecting its detached runtime."""

    sys.path.insert(0, str(root))
    from experiments.c0c3_factorial import codex_cli, evaluator, runner, state
    from experiments.c0c3_factorial.cli import _load_campaign
    from experiments.c0c3_factorial.orchestration import (
        FACTORIAL_STAGE,
        _append_trajectory_lifecycle,
        _freeze_artifact_clean_assumption_prompt,
        _pause_request_path,
        _take_pause_request,
        trajectory_lock,
    )
    def load_guard(name: str, filename: str) -> Any:
        path = REPO_ROOT / "experiments" / filename
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"cannot load compatibility guard: {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
        return module

    provider_guard = load_guard("rl4rl_c0c3_provider_guard", "c0c3_provider_guard.py")
    provider_guard.install_provider_retry_guard(codex_cli)
    duplicate_guard = load_guard(
        "rl4rl_c0c3_duplicate_guard", "c0c3_duplicate_guard.py"
    )
    duplicate_guard.install_duplicate_guard(runner, state, evaluator)
    c4_guard = load_guard("rl4rl_c0c3_v21_c4_guard", "c0c3_v21_c4_guard.py")
    c4_guard.install_v21_c4_guard(runner, state)

    shared_capacity = os.environ.get(SHARED_LOCAL_EVALUATOR_CAPACITY_ENV)
    if shared_capacity is not None:
        evaluator.SHARED_LOCAL_EVALUATOR_CAPACITY = int(shared_capacity)
    campaign_capacity = os.environ.get(CAMPAIGN_LOCAL_EVALUATOR_CAPACITY_ENV)
    if campaign_capacity is not None:
        original_make_evaluator = runner.make_command_evaluator
        runner.make_command_evaluator = lambda **kwargs: original_make_evaluator(
            **(kwargs | {"max_parallel_evaluators": int(campaign_capacity)})
        )

    return {
        "_load_campaign": _load_campaign,
        "FACTORIAL_STAGE": FACTORIAL_STAGE,
        "_append_trajectory_lifecycle": _append_trajectory_lifecycle,
        "_freeze_artifact_clean_assumption_prompt": (
            _freeze_artifact_clean_assumption_prompt
        ),
        "_pause_request_path": _pause_request_path,
        "_take_pause_request": _take_pause_request,
        "trajectory_lock": trajectory_lock,
        "run_one_opportunity": runner.run_one_opportunity,
        "SearchController": state.SearchController,
        "atomic_json": state.atomic_json,
        "utc_now": state.utc_now,
    }


def _assignment(
    campaign: Path, run_id: str, *, schedule_file: str
) -> dict[str, object]:
    schedule = json.loads((campaign / schedule_file).read_text(encoding="utf-8"))
    if not isinstance(schedule, list):
        raise ValueError("campaign schedule must be a list")
    match = next(
        (
            row
            for row in schedule
            if isinstance(row, dict) and str(row.get("run_id")) == run_id
        ),
        None,
    )
    if match is None:
        raise ValueError(f"run ID is not in the campaign schedule: {run_id}")
    return dict(match)


def run_extension_trajectory(
    campaign_dir: str | Path,
    *,
    runtime_root: str | Path,
    run_id: str,
    python_bin: str,
    codex_binary: str = "codex",
    codex_timeout_seconds: int = 3600,
    resume: bool = False,
    schedule_file: str = "schedule.json",
) -> dict[str, object]:
    """Run one appended C0-C3 trajectory until completion or a safe pause."""

    campaign = Path(campaign_dir).expanduser().resolve()
    runtime = _load_runtime(Path(runtime_root).expanduser().resolve())
    spec, task, framework = runtime["_load_campaign"](campaign)
    assignment = _assignment(campaign, run_id, schedule_file=schedule_file)
    block = int(assignment["block"])
    condition = str(assignment["condition"])
    if block <= spec.blocks and condition != "C4":
        raise ValueError(
            "extension controller is only for a block appended beyond the base protocol"
        )
    if condition not in {"C0", "C1", "C2", "C3", "C4"}:
        raise ValueError("extension controller supports factorial C0-C4 runs only")
    if condition == "C4" and spec.protocol_version != "2.1":
        raise ValueError("C4 extension requires protocol 2.1")
    if not spec.c0c3_only:
        raise ValueError("extension controller requires a C0-C3-only campaign")

    run_dir = campaign / "runs" / run_id
    stage = runtime["FACTORIAL_STAGE"]
    SearchController = runtime["SearchController"]
    append_lifecycle = runtime["_append_trajectory_lifecycle"]
    with runtime["trajectory_lock"](run_dir):
        controller = SearchController.load(run_dir, spec)
        if controller.state.active is not None:
            raise RuntimeError(
                "run has an interrupted active opportunity; recover it explicitly "
                "before starting or resuming"
            )
        if controller.state.status == "completed":
            return {
                "run_id": run_id,
                "condition": str(assignment["condition"]),
                "status": "completed",
                "proposals_used": controller.state.proposals_used,
                "completed_opportunities": 0,
                "stop_reason": "already_completed",
            }
        if not resume and controller.state.proposals_used != 0:
            raise ValueError(
                "an already-started extension trajectory must use --resume"
            )
        if resume:
            cleared = runtime["_take_pause_request"](run_dir)
            if cleared is not None:
                append_lifecycle(
                    campaign,
                    run_dir,
                    event="trajectory_pause_cleared_for_resume",
                    assignment=assignment,
                    stage=stage,
                    prior_reason=str(cleared.get("reason", "[not recorded]")),
                )
        elif runtime["_pause_request_path"](run_dir).exists():
            raise RuntimeError("a pause request is pending; use --resume to clear it")

        prompt_snapshot = (
            runtime["_freeze_artifact_clean_assumption_prompt"](
                campaign=campaign,
                run_dir=run_dir,
                repo_root=Path(runtime_root).expanduser().resolve(),
                framework=framework,
            )
            if controller.state.proposals_used == 0
            else None
        )
        append_lifecycle(
            campaign,
            run_dir,
            event="trajectory_resumed" if resume else "trajectory_started",
            assignment=assignment,
            stage=stage,
            starting_opportunity=controller.state.next_opportunity,
            assumption_prompt_sha256=(
                prompt_snapshot.get("sha256") if prompt_snapshot is not None else None
            ),
            controller_kind="appended-block-extension",
        )

        completed_opportunities = 0
        while True:
            controller = SearchController.load(run_dir, spec)
            if controller.state.active is not None:
                raise RuntimeError(
                    "run has an interrupted active opportunity; recover it explicitly"
                )
            if controller.state.status == "completed":
                outcome = {
                    "run_id": run_id,
                    "condition": str(assignment["condition"]),
                    "status": "completed",
                    "proposals_used": controller.state.proposals_used,
                    "completed_opportunities": completed_opportunities,
                    "stop_reason": "budget_completed",
                }
                append_lifecycle(
                    campaign,
                    run_dir,
                    event="trajectory_completed",
                    assignment=assignment,
                    stage=stage,
                    **outcome,
                )
                return outcome
            record = runtime["run_one_opportunity"](
                run_dir,
                spec=spec,
                task=task,
                framework=framework,
                repo_root=Path(runtime_root).expanduser().resolve(),
                python_bin=python_bin,
                codex_binary=codex_binary,
                codex_timeout_seconds=codex_timeout_seconds,
            )
            completed_opportunities += 1
            if (
                record.get("evaluation", {}).get("failure_kind") == "provider"
                and record.get("usage_increment", {}).get("total_tokens", 0) == 0
            ):
                stopped = SearchController.load(run_dir, spec).state
                outcome = {
                    "run_id": run_id,
                    "condition": str(assignment["condition"]),
                    "status": stopped.status,
                    "proposals_used": stopped.proposals_used,
                    "completed_opportunities": completed_opportunities,
                    "stop_reason": "provider_transport_failure",
                }
                append_lifecycle(
                    campaign,
                    run_dir,
                    event="trajectory_stopped_provider_failure",
                    assignment=assignment,
                    stage=stage,
                    **outcome,
                )
                return outcome
            pause_request = runtime["_take_pause_request"](run_dir)
            if pause_request is not None:
                outcome = {
                    "run_id": run_id,
                    "condition": str(assignment["condition"]),
                    "status": "paused",
                    "proposals_used": controller.state.proposals_used,
                    "completed_opportunities": completed_opportunities,
                    "stop_reason": "cooperative_pause",
                }
                append_lifecycle(
                    campaign,
                    run_dir,
                    event="trajectory_paused",
                    assignment=assignment,
                    stage=stage,
                    reason=str(pause_request.get("reason", "[not recorded]")),
                    **outcome,
                )
                return outcome


def request_extension_pause(
    campaign_dir: str | Path,
    *,
    runtime_root: str | Path,
    run_id: str,
    schedule_file: str,
    reason: str,
) -> dict[str, object]:
    """Request a boundary-safe pause for an appended or C4 trajectory."""

    if not reason.strip():
        raise ValueError("pause reason cannot be blank")
    campaign = Path(campaign_dir).expanduser().resolve()
    runtime = _load_runtime(Path(runtime_root).expanduser().resolve())
    assignment = _assignment(campaign, run_id, schedule_file=schedule_file)
    run_dir = campaign / "runs" / run_id
    controller = runtime["SearchController"].load(
        run_dir, runtime["_load_campaign"](campaign)[0]
    )
    request = {
        "schema_version": "1.0",
        "run_id": run_id,
        "requested_at": runtime["utc_now"](),
        "reason": reason.strip(),
    }
    runtime["atomic_json"](runtime["_pause_request_path"](run_dir), request)
    runtime["_append_trajectory_lifecycle"](
        campaign,
        run_dir,
        event="trajectory_pause_requested",
        assignment=assignment,
        stage=runtime["FACTORIAL_STAGE"],
        reason=reason.strip(),
        active_opportunity=(
            controller.state.active.index
            if controller.state.active is not None
            else None
        ),
    )
    return {
        "run_id": run_id,
        "status": "pause_requested",
        "active_opportunity": (
            controller.state.active.index
            if controller.state.active is not None
            else None
        ),
    }
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--codex-timeout", type=int, default=3600)
    parser.add_argument("--schedule-file", default="schedule.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--reason", default="operator requested cooperative pause")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    print(
        json.dumps(
            (
                request_extension_pause(
                    args.campaign,
                    runtime_root=args.runtime_root,
                    run_id=args.run_id,
                    schedule_file=args.schedule_file,
                    reason=args.reason,
                )
                if args.pause
                else run_extension_trajectory(
                    args.campaign,
                    runtime_root=args.runtime_root,
                    run_id=args.run_id,
                    python_bin=args.python_bin,
                    codex_binary=args.codex_binary,
                    codex_timeout_seconds=args.codex_timeout,
                    resume=args.resume,
                    schedule_file=args.schedule_file,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

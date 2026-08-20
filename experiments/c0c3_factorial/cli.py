#!/usr/bin/env python3
"""Create, validate, and run controlled C0-C3 campaigns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .campaign import calibrate_task, create_campaign
from .runner import run_one_opportunity
from .spec import (
    BudgetSpec,
    ExecutionBackend,
    FactorialSpec,
    FrameworkKind,
    FrameworkSpec,
    ModelSpec,
    ObjectiveDirection,
    TaskSpec,
)
from .state import SearchController

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]


def _load_campaign(campaign: Path) -> tuple[FactorialSpec, TaskSpec, FrameworkSpec]:
    inputs = campaign / "inputs"
    protocol = json.loads((inputs / "protocol.json").read_text(encoding="utf-8"))
    model = ModelSpec(**protocol.pop("model"))
    budget = BudgetSpec(**protocol.pop("budget"))
    protocol["transition_opportunities"] = tuple(protocol["transition_opportunities"])
    spec = FactorialSpec(**protocol, model=model, budget=budget)
    task_payload = json.loads((inputs / "task.json").read_text(encoding="utf-8"))
    for key in (
        "editable_paths",
        "evaluator_command",
        "public_feedback_metrics",
        "final_holdout_command",
    ):
        task_payload[key] = tuple(task_payload[key])
    task_payload["objective_direction"] = ObjectiveDirection(
        task_payload["objective_direction"]
    )
    task_payload["preferred_backend"] = ExecutionBackend(
        task_payload["preferred_backend"]
    )
    task = TaskSpec(**task_payload)
    framework_payload = json.loads(
        (inputs / "framework.json").read_text(encoding="utf-8")
    )
    framework_payload["framework_id"] = FrameworkKind(framework_payload["framework_id"])
    framework = FrameworkSpec(**framework_payload)
    return spec, task, framework


def command_calibrate(args: argparse.Namespace) -> int:
    spec = FactorialSpec.from_toml(args.protocol)
    task = TaskSpec.from_toml(args.task)
    path = calibrate_task(
        args.output,
        spec=spec,
        task=task,
        repo_root=REPO_ROOT,
        python_bin=args.python_bin,
    )
    print(path)
    return 0


def command_create(args: argparse.Namespace) -> int:
    spec = FactorialSpec.from_toml(args.protocol)
    task = TaskSpec.from_toml(args.task)
    framework = FrameworkSpec.from_toml(args.framework)
    output = create_campaign(
        args.output,
        spec=spec,
        task=task,
        framework=framework,
        calibration_path=args.baseline,
        repo_root=REPO_ROOT,
        include_no_search=not args.without_no_search,
    )
    print(output)
    return 0


def _run_once(args: argparse.Namespace) -> dict[str, object]:
    campaign = args.campaign.resolve()
    spec, task, framework = _load_campaign(campaign)
    run_dir = campaign / "runs" / args.run_id
    return run_one_opportunity(
        run_dir,
        spec=spec,
        task=task,
        framework=framework,
        repo_root=REPO_ROOT,
        python_bin=args.python_bin,
        codex_binary=args.codex_binary,
        codex_timeout_seconds=args.codex_timeout,
    )


def command_run_one(args: argparse.Namespace) -> int:
    record = _run_once(args)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def command_run(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, _task, _framework = _load_campaign(campaign)
    while True:
        controller = SearchController.load(campaign / "runs" / args.run_id, spec)
        if controller.state.status == "completed":
            break
        record = _run_once(args)
        print(
            f"{record['run_id']} opportunity={record['opportunity']} "
            f"retained={record['retained']} "
            f"decision={record['retention_decision']}",
            flush=True,
        )
    return 0


def command_status(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, _task, _framework = _load_campaign(campaign)
    schedule = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    for assignment in schedule:
        controller = SearchController.load(
            campaign / "runs" / assignment["run_id"], spec
        )
        remaining = controller.remaining()
        print(
            "\t".join(
                (
                    str(assignment["run_id"]),
                    str(assignment["condition"]),
                    controller.state.status,
                    str(controller.state.proposals_used),
                    str(controller.state.evaluations_used),
                    str(controller.state.usage.total_tokens),
                    f"{float(remaining['evaluator_seconds']):.1f}",
                )
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--protocol", type=Path, required=True)
    calibrate.add_argument("--task", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--python-bin", default=sys.executable)
    calibrate.set_defaults(handler=command_calibrate)

    create = subparsers.add_parser("create")
    create.add_argument("--protocol", type=Path, required=True)
    create.add_argument("--task", type=Path, required=True)
    create.add_argument("--framework", type=Path, required=True)
    create.add_argument("--baseline", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--without-no-search", action="store_true")
    create.set_defaults(handler=command_create)

    for name, handler in (("run-one", command_run_one), ("run", command_run)):
        run = subparsers.add_parser(name)
        run.add_argument("--campaign", type=Path, required=True)
        run.add_argument("--run-id", required=True)
        run.add_argument("--python-bin", default=sys.executable)
        run.add_argument("--codex-binary", default="codex")
        run.add_argument("--codex-timeout", type=int, default=3600)
        run.set_defaults(handler=handler)

    status = subparsers.add_parser("status")
    status.add_argument("--campaign", type=Path, required=True)
    status.set_defaults(handler=command_status)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

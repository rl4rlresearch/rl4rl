#!/usr/bin/env python3
"""Create, validate, and run controlled C0-C3 campaigns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .campaign import calibrate_task, create_campaign
from .orchestration import next_run
from .postsearch import export_layer_b_packets, run_layer_c, score_layer_b
from .runner import recover_active_opportunity, run_one_opportunity
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
from .validation import validate_campaign

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


def _run_once(
    args: argparse.Namespace, *, run_id: str | None = None
) -> dict[str, object]:
    campaign = args.campaign.resolve()
    spec, task, framework = _load_campaign(campaign)
    selected_run_id = run_id or args.run_id
    run_dir = campaign / "runs" / selected_run_id
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


def command_run_next(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, _task, _framework = _load_campaign(campaign)
    selected = next_run(campaign, spec)
    if selected is None:
        print("campaign completed")
        return 0
    record = _run_once(args, run_id=selected.run_id)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def command_run_campaign(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, _task, _framework = _load_campaign(campaign)
    completed = 0
    while args.max_opportunities is None or completed < args.max_opportunities:
        selected = next_run(campaign, spec)
        if selected is None:
            print("campaign completed", flush=True)
            break
        record = _run_once(args, run_id=selected.run_id)
        completed += 1
        print(
            f"{record['run_id']} opportunity={record['opportunity']} "
            f"condition={record['condition']} retained={record['retained']} "
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


def command_validate(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, task, framework = _load_campaign(campaign)
    report = validate_campaign(
        campaign,
        spec=spec,
        task=task,
        framework=framework,
        repo_root=REPO_ROOT,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


def command_export_layer_b(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, task, _framework = _load_campaign(campaign)
    print(export_layer_b_packets(campaign, spec=spec, task=task))
    return 0


def command_score_layer_b(args: argparse.Namespace) -> int:
    print(
        score_layer_b(
            args.campaign.resolve(), annotations_path=args.annotations.resolve()
        )
    )
    return 0


def command_layer_c(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, task, _framework = _load_campaign(campaign)
    print(
        run_layer_c(
            campaign,
            spec=spec,
            task=task,
            repo_root=REPO_ROOT,
            python_bin=args.python_bin,
        )
    )
    return 0


def command_recover(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, _task, _framework = _load_campaign(campaign)
    record = recover_active_opportunity(
        campaign / "runs" / args.run_id,
        spec=spec,
        reason=args.reason,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
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

    for name, handler in (
        ("run-next", command_run_next),
        ("run-campaign", command_run_campaign),
    ):
        run = subparsers.add_parser(name)
        run.add_argument("--campaign", type=Path, required=True)
        run.add_argument("--python-bin", default=sys.executable)
        run.add_argument("--codex-binary", default="codex")
        run.add_argument("--codex-timeout", type=int, default=3600)
        if name == "run-campaign":
            run.add_argument("--max-opportunities", type=int)
        run.set_defaults(handler=handler)

    status = subparsers.add_parser("status")
    status.add_argument("--campaign", type=Path, required=True)
    status.set_defaults(handler=command_status)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--campaign", type=Path, required=True)
    validate.set_defaults(handler=command_validate)

    export_layer_b = subparsers.add_parser("export-layer-b")
    export_layer_b.add_argument("--campaign", type=Path, required=True)
    export_layer_b.set_defaults(handler=command_export_layer_b)

    score = subparsers.add_parser("score-layer-b")
    score.add_argument("--campaign", type=Path, required=True)
    score.add_argument("--annotations", type=Path, required=True)
    score.set_defaults(handler=command_score_layer_b)

    layer_c = subparsers.add_parser("run-layer-c")
    layer_c.add_argument("--campaign", type=Path, required=True)
    layer_c.add_argument("--python-bin", default=sys.executable)
    layer_c.set_defaults(handler=command_layer_c)

    recover = subparsers.add_parser("recover-active")
    recover.add_argument("--campaign", type=Path, required=True)
    recover.add_argument("--run-id", required=True)
    recover.add_argument("--reason", required=True)
    recover.set_defaults(handler=command_recover)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

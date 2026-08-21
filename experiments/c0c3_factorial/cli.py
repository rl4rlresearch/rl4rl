#!/usr/bin/env python3
"""Create, validate, and run controlled C0-C3 campaigns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .campaign import (
    calibrate_task,
    create_campaign,
    execute_calibration,
    prepare_calibration,
)
from .orchestration import (
    STAGED_EXECUTION_STAGES,
    campaign_lock,
    next_run,
    run_parallel_campaign,
    run_parallel_next,
    run_staged_campaign,
    run_staged_independent_campaign,
    run_staged_next,
)
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


def _load_spec(path: Path) -> FactorialSpec:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    model = ModelSpec(**protocol.pop("model"))
    budget = BudgetSpec(**protocol.pop("budget"))
    protocol["transition_opportunities"] = tuple(protocol["transition_opportunities"])
    return FactorialSpec(**protocol, model=model, budget=budget)


def _load_task(path: Path) -> TaskSpec:
    task_payload = json.loads(path.read_text(encoding="utf-8"))
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
    return TaskSpec(**task_payload)


def _load_campaign(campaign: Path) -> tuple[FactorialSpec, TaskSpec, FrameworkSpec]:
    inputs = campaign / "inputs"
    spec = _load_spec(inputs / "protocol.json")
    task = _load_task(inputs / "task.json")
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


def command_prepare_calibration(args: argparse.Namespace) -> int:
    spec = FactorialSpec.from_toml(args.protocol)
    task = TaskSpec.from_toml(args.task)
    print(
        prepare_calibration(
            args.output,
            spec=spec,
            task=task,
            repo_root=REPO_ROOT,
        )
    )
    return 0


def command_execute_calibration(args: argparse.Namespace) -> int:
    calibration = args.calibration.resolve()
    spec = _load_spec(calibration / "inputs/protocol.json")
    task = _load_task(calibration / "inputs/task.json")
    print(
        execute_calibration(
            calibration,
            spec=spec,
            task=task,
            repo_root=REPO_ROOT,
            python_bin=args.python_bin,
        )
    )
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
    with campaign_lock(args.campaign):
        record = _run_once(args)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def command_run(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, _task, _framework = _load_campaign(campaign)
    with campaign_lock(campaign):
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
    with campaign_lock(campaign):
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
    with campaign_lock(campaign):
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


def command_run_parallel_next(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, task, framework = _load_campaign(campaign)
    result = run_parallel_next(
        campaign,
        spec=spec,
        task=task,
        framework=framework,
        repo_root=REPO_ROOT,
        python_bin=args.python_bin,
        codex_binary=args.codex_binary,
        codex_timeout_seconds=args.codex_timeout,
    )
    if result is None:
        print("campaign completed")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_run_parallel_campaign(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, task, framework = _load_campaign(campaign)
    completed = 0
    for result in run_parallel_campaign(
        campaign,
        spec=spec,
        task=task,
        framework=framework,
        repo_root=REPO_ROOT,
        python_bin=args.python_bin,
        codex_binary=args.codex_binary,
        codex_timeout_seconds=args.codex_timeout,
        max_block_rounds=args.max_block_rounds,
    ):
        completed += 1
        factorial = ",".join(
            str(record["condition"])
            for record in result["factorial_records"]
        )
        print(
            f"block={result['block']} opportunity={result['opportunity']} "
            f"parallel_conditions={factorial or '-'} "
            f"n0={'yes' if result['no_search_record'] is not None else 'no'}",
            flush=True,
        )
    if completed == 0:
        print("campaign completed", flush=True)
    return 0


def command_run_staged_next(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, task, framework = _load_campaign(campaign)
    result = run_staged_next(
        campaign,
        spec=spec,
        task=task,
        framework=framework,
        repo_root=REPO_ROOT,
        python_bin=args.python_bin,
        block=args.block,
        stage=args.stage,
        codex_binary=args.codex_binary,
        codex_timeout_seconds=args.codex_timeout,
    )
    if result is None:
        print(f"block {args.block} stage {args.stage} completed")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_run_staged_campaign(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, task, framework = _load_campaign(campaign)
    completed = 0
    for result in run_staged_campaign(
        campaign,
        spec=spec,
        task=task,
        framework=framework,
        repo_root=REPO_ROOT,
        python_bin=args.python_bin,
        block=args.block,
        stage=args.stage,
        codex_binary=args.codex_binary,
        codex_timeout_seconds=args.codex_timeout,
        max_block_rounds=args.max_block_rounds,
    ):
        completed += 1
        factorial = ",".join(
            str(record["condition"]) for record in result["factorial_records"]
        )
        print(
            f"stage={result['execution_stage']} "
            f"opportunity={result['opportunity']} "
            f"parallel_conditions={factorial or '-'} "
            f"n0={'yes' if result['no_search_record'] is not None else 'no'}",
            flush=True,
        )
    if completed == 0:
        print(f"block {args.block} stage {args.stage} completed", flush=True)
    return 0


def command_run_staged_independent_campaign(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    spec, task, framework = _load_campaign(campaign)
    completed = 0
    for result in run_staged_independent_campaign(
        campaign,
        spec=spec,
        task=task,
        framework=framework,
        repo_root=REPO_ROOT,
        python_bin=args.python_bin,
        block=args.block,
        stage=args.stage,
        codex_binary=args.codex_binary,
        codex_timeout_seconds=args.codex_timeout,
    ):
        completed += 1
        print(
            f"stage={result['execution_stage']} "
            f"run_id={result['run_id']} "
            f"condition={result['condition']} "
            f"proposals_used={result['proposals_used']} "
            f"status={result['status']}",
            flush=True,
        )
    if completed == 0:
        print(f"block {args.block} stage {args.stage} completed", flush=True)
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
    with campaign_lock(campaign):
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

    prepare_calibration_parser = subparsers.add_parser("prepare-calibration")
    prepare_calibration_parser.add_argument("--protocol", type=Path, required=True)
    prepare_calibration_parser.add_argument("--task", type=Path, required=True)
    prepare_calibration_parser.add_argument("--output", type=Path, required=True)
    prepare_calibration_parser.set_defaults(handler=command_prepare_calibration)

    execute_calibration_parser = subparsers.add_parser("execute-calibration")
    execute_calibration_parser.add_argument(
        "--calibration", type=Path, required=True
    )
    execute_calibration_parser.add_argument("--python-bin", default=sys.executable)
    execute_calibration_parser.set_defaults(handler=command_execute_calibration)

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

    for name, handler in (
        ("run-parallel-next", command_run_parallel_next),
        ("run-parallel-campaign", command_run_parallel_campaign),
    ):
        run = subparsers.add_parser(name)
        run.add_argument("--campaign", type=Path, required=True)
        run.add_argument("--python-bin", default=sys.executable)
        run.add_argument("--codex-binary", default="codex")
        run.add_argument("--codex-timeout", type=int, default=3600)
        if name == "run-parallel-campaign":
            run.add_argument("--max-block-rounds", type=int)
        run.set_defaults(handler=handler)

    for name, handler in (
        ("run-staged-next", command_run_staged_next),
        ("run-staged-campaign", command_run_staged_campaign),
    ):
        run = subparsers.add_parser(name)
        run.add_argument("--campaign", type=Path, required=True)
        run.add_argument("--block", type=int, required=True)
        run.add_argument(
            "--stage",
            choices=sorted(STAGED_EXECUTION_STAGES),
            required=True,
        )
        run.add_argument("--python-bin", default=sys.executable)
        run.add_argument("--codex-binary", default="codex")
        run.add_argument("--codex-timeout", type=int, default=3600)
        if name == "run-staged-campaign":
            run.add_argument("--max-block-rounds", type=int)
        run.set_defaults(handler=handler)

    staged_independent = subparsers.add_parser("run-staged-independent-campaign")
    staged_independent.add_argument("--campaign", type=Path, required=True)
    staged_independent.add_argument("--block", type=int, required=True)
    staged_independent.add_argument(
        "--stage",
        choices=sorted(STAGED_EXECUTION_STAGES),
        required=True,
    )
    staged_independent.add_argument("--python-bin", default=sys.executable)
    staged_independent.add_argument("--codex-binary", default="codex")
    staged_independent.add_argument("--codex-timeout", type=int, default=3600)
    staged_independent.set_defaults(handler=command_run_staged_independent_campaign)

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

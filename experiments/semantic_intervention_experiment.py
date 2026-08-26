#!/usr/bin/env python3
"""Prepare, validate, run, and control semantic intervention campaigns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c0c3_factorial.campaign import calibrate_task  # noqa: E402
from experiments.c0c3_factorial.semantic_interventions import (  # noqa: E402
    create_semantic_campaign,
    run_semantic_campaign,
    run_semantic_opportunity,
    semantic_status,
    set_semantic_control,
    set_semantic_run_control,
    validate_semantic_campaign,
)
from experiments.c0c3_factorial.spec import (  # noqa: E402
    FactorialSpec,
    FrameworkSpec,
    TaskSpec,
)


def _inputs(args: argparse.Namespace):
    return (
        FactorialSpec.from_toml(args.protocol),
        TaskSpec.from_toml(args.task),
        FrameworkSpec.from_toml(args.framework),
    )


def _python_bin(path: Path) -> str:
    """Return an absolute interpreter path without resolving venv symlinks."""

    return str(path.absolute())


def command_prepare(args: argparse.Namespace) -> int:
    spec, task, framework = _inputs(args)
    calibration = args.calibration.resolve()
    baseline = calibration / "baseline.json"
    if not baseline.is_file():
        calibrate_task(
            calibration,
            spec=spec,
            task=task,
            repo_root=args.repo_root.resolve(),
            python_bin=_python_bin(args.python_bin),
        )
    campaign = create_semantic_campaign(
        args.output,
        spec=spec,
        task=task,
        framework=framework,
        intervention_plan_path=args.interventions,
        calibration_path=baseline,
        repo_root=args.repo_root.resolve(),
    )
    validation = validate_semantic_campaign(
        campaign, repo_root=args.repo_root.resolve()
    )
    print(json.dumps({"campaign": str(campaign), "validation": validation}, indent=2))
    return 0 if validation["valid"] else 1


def command_validate(args: argparse.Namespace) -> int:
    result = validate_semantic_campaign(
        args.campaign, repo_root=args.repo_root.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_run_one(args: argparse.Namespace) -> int:
    result = run_semantic_opportunity(
        args.campaign,
        run_id=args.run_id,
        repo_root=args.repo_root.resolve(),
        python_bin=_python_bin(args.python_bin),
        codex_binary=args.codex_binary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_run_campaign(args: argparse.Namespace) -> int:
    result = run_semantic_campaign(
        args.campaign,
        repo_root=args.repo_root.resolve(),
        python_bin=_python_bin(args.python_bin),
        max_workers=args.max_workers,
        recover_interrupted=args.recover_interrupted,
        codex_binary=args.codex_binary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_status(args: argparse.Namespace) -> int:
    print(json.dumps(semantic_status(args.campaign), indent=2, sort_keys=True))
    return 0


def command_control(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            set_semantic_control(
                args.campaign, desired=args.desired, reason=args.reason
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_control_run(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            set_semantic_run_control(
                args.campaign,
                run_id=args.run_id,
                desired=args.desired,
                reason=args.reason,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--task", type=Path, required=True)
    prepare.add_argument("--framework", type=Path, required=True)
    prepare.add_argument("--interventions", type=Path, required=True)
    prepare.add_argument("--calibration", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    prepare.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    prepare.set_defaults(handler=command_prepare)

    validate = sub.add_parser("validate")
    validate.add_argument("--campaign", type=Path, required=True)
    validate.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    validate.set_defaults(handler=command_validate)

    run_one = sub.add_parser("run-one")
    run_one.add_argument("--campaign", type=Path, required=True)
    run_one.add_argument("--run-id", required=True)
    run_one.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    run_one.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    run_one.add_argument("--codex-binary", default="codex")
    run_one.set_defaults(handler=command_run_one)

    run_campaign = sub.add_parser("run-campaign")
    run_campaign.add_argument("--campaign", type=Path, required=True)
    run_campaign.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    run_campaign.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    run_campaign.add_argument("--codex-binary", default="codex")
    run_campaign.add_argument("--max-workers", type=int)
    run_campaign.add_argument("--recover-interrupted", action="store_true")
    run_campaign.set_defaults(handler=command_run_campaign)

    status = sub.add_parser("status")
    status.add_argument("--campaign", type=Path, required=True)
    status.set_defaults(handler=command_status)

    control = sub.add_parser("control")
    control.add_argument("--campaign", type=Path, required=True)
    control.add_argument(
        "--desired", choices=("running", "paused", "stopped"), required=True
    )
    control.add_argument("--reason", required=True)
    control.set_defaults(handler=command_control)
    control_run = sub.add_parser("control-run")
    control_run.add_argument("--campaign", type=Path, required=True)
    control_run.add_argument("--run-id", required=True)
    control_run.add_argument(
        "--desired", choices=("running", "paused", "stopped"), required=True
    )
    control_run.add_argument("--reason", required=True)
    control_run.set_defaults(handler=command_control_run)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

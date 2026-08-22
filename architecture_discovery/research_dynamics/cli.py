"""Command line interface for planning, running, and exporting experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_dynamics.annotations import export_blinded_annotations
from research_dynamics.contracts import (
    ConditionId,
    FrameworkKind,
    ProcessCondition,
    ProcessStudyConfig,
)
from research_dynamics.extraction import export_run
from research_dynamics.memory import read_jsonl
from research_dynamics.metrics import summarize_annotations, summarize_decisions
from research_dynamics.orchestration import (
    execute_manifest,
    plan_forks,
    plan_full_trajectories,
)
from study.serialization import atomic_write_json


def _command(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("command must be a JSON argv array") from error
    if not isinstance(parsed, list) or not parsed or any(
        not isinstance(item, str) or not item for item in parsed
    ):
        raise argparse.ArgumentTypeError("command must be a non-empty JSON string array")
    return parsed


def _schedule(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("schedule must be comma-separated integers") from error
    if any(item < 1 for item in parsed) or tuple(sorted(set(parsed))) != parsed:
        raise argparse.ArgumentTypeError("schedule must be positive, sorted, and unique")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-process")
    sub = parser.add_subparsers(dest="action", required=True)

    forks = sub.add_parser("plan-forks", help="create a matched four-cell checkpoint fork")
    forks.add_argument("--study-id", required=True)
    forks.add_argument("--framework", type=FrameworkKind, required=True)
    forks.add_argument("--checkpoint", type=Path, required=True)
    forks.add_argument("--output-dir", type=Path, required=True)
    forks.add_argument("--command-json", type=_command, required=True)
    forks.add_argument("--horizon", type=int, default=4)
    forks.add_argument("--seed", type=int, default=1)
    forks.add_argument("--scientific", action="store_true")

    full = sub.add_parser("plan-full", help="create block-randomized full trajectories")
    full.add_argument("--study-id", required=True)
    full.add_argument("--framework", type=FrameworkKind, required=True)
    full.add_argument("--output-dir", type=Path, required=True)
    full.add_argument("--command-json", type=_command, required=True)
    full.add_argument("--blocks", type=int, required=True)
    full.add_argument(
        "--iterations",
        type=_positive_int,
        required=True,
        help="proposal opportunities per trajectory; available as {iterations} in command-json",
    )
    full.add_argument("--first-seed", type=int, default=1)
    full.add_argument("--challenge-schedule", type=_schedule, default=(5, 10, 15, 20))
    full.add_argument("--scientific", action="store_true")

    run = sub.add_parser("run-manifest", help="execute randomized branches sequentially")
    run.add_argument("manifest", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--continue-on-failure", action="store_true")

    extract = sub.add_parser("extract", help="normalize one completed controller run")
    extract.add_argument("--run-dir", type=Path, required=True)
    extract.add_argument("--config", type=Path)

    baseline = sub.add_parser(
        "import-baseline",
        help="reconstruct a pre-intervention run without inventing missing notes",
    )
    baseline.add_argument("--run-dir", type=Path, required=True)
    baseline.add_argument("--framework", type=FrameworkKind, required=True)
    baseline.add_argument("--study-id", required=True)
    baseline.add_argument("--run-id", required=True)

    blind = sub.add_parser("export-annotations", help="make blinded review packets")
    blind.add_argument("--decisions", type=Path, required=True)
    blind.add_argument("--output-dir", type=Path, required=True)

    summary = sub.add_parser("summarize", help="summarize completed annotations")
    summary.add_argument("--annotations", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)

    telemetry = sub.add_parser(
        "summarize-decisions",
        help="compute annotation-free idea and lineage diagnostics",
    )
    telemetry.add_argument("--decisions", type=Path, required=True)
    telemetry.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "plan-forks":
        path = plan_forks(
            study_id=args.study_id,
            framework=args.framework,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            command=args.command_json,
            horizon=args.horizon,
            seed=args.seed,
            scientific=args.scientific,
        )
        print(path)
    elif args.action == "plan-full":
        path = plan_full_trajectories(
            study_id=args.study_id,
            framework=args.framework,
            output_dir=args.output_dir,
            command=args.command_json,
            blocks=args.blocks,
            iterations=args.iterations,
            first_seed=args.first_seed,
            challenge_schedule=args.challenge_schedule,
            scientific=args.scientific,
        )
        print(path)
    elif args.action == "run-manifest":
        print(
            json.dumps(
                execute_manifest(
                    args.manifest,
                    dry_run=args.dry_run,
                    continue_on_failure=args.continue_on_failure,
                ),
                indent=2,
            )
        )
    elif args.action == "extract":
        config_path = args.config or args.run_dir / "research_process" / "study_config.json"
        config = ProcessStudyConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
        print(export_run(args.run_dir, config))
    elif args.action == "import-baseline":
        config = ProcessStudyConfig(
            study_id=args.study_id,
            run_id=args.run_id,
            framework=args.framework,
            condition=ProcessCondition.for_id(ConditionId.RD0),
            challenge_opportunities=(),
            scientific=False,
        )
        config_path = args.run_dir / "research_process" / "study_config.json"
        atomic_write_json(config_path, config.to_dict())
        print(export_run(args.run_dir, config))
    elif args.action == "export-annotations":
        print("\n".join(str(path) for path in export_blinded_annotations(args.decisions, args.output_dir)))
    elif args.action == "summarize":
        result = summarize_annotations(read_jsonl(args.annotations))
        atomic_write_json(args.output, result)
        print(args.output)
    elif args.action == "summarize-decisions":
        result = summarize_decisions(read_jsonl(args.decisions))
        atomic_write_json(args.output, result)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

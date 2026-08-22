#!/usr/bin/env python3
"""Trusted task wrappers that emit the shared Layer A JSON contract."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ACCURACY = re.compile(r"Results: (\d+)/(\d+) correct \(([0-9.]+)%\)")
PARAMETERS = re.compile(r"Parameters \(unique\):\s*(\d+)")
TRAINING_STEP = re.compile(r"\bstep\s+(\d+)\b")


def _run(command: list[str], *, cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(completed.stdout, end="")
    return completed.returncode, completed.stdout


def evaluate_adderboard(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    train_output = ""
    if not args.verify_existing_checkpoint:
        train_exit, train_output = _run(
            [args.python_bin, "src/train.py"], cwd=workspace
        )
        if train_exit:
            return train_exit
    verify_exit, verify_output = _run(
        [
            args.python_bin,
            str(args.repo_root / "architecture_discovery/vendor/AdderBoard/verify.py"),
            str(workspace / "submission.py"),
            "--num-tests",
            str(args.num_tests),
            "--seed",
            str(args.seed),
        ],
        cwd=workspace,
    )
    accuracy_match = ACCURACY.search(verify_output)
    parameters_match = PARAMETERS.search(verify_output)
    if verify_exit or accuracy_match is None or parameters_match is None:
        return verify_exit or 2
    steps = [int(value) for value in TRAINING_STEP.findall(train_output)]
    payload = {
        "schema_version": "1.0",
        "layer": args.layer,
        "metrics": {
            "accuracy": float(accuracy_match.group(3)) / 100.0,
            "parameters": int(parameters_match.group(1)),
            "correct": int(accuracy_match.group(1)),
            "cases": int(accuracy_match.group(2)),
            "training_steps": max(steps, default=0),
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, layer, default_seed in (
        ("adderboard", "A", 2025),
        ("adderboard-holdout", "C", 8_724_319),
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--workspace", type=Path, required=True)
        subparser.add_argument("--repo-root", type=Path, required=True)
        subparser.add_argument("--python-bin", default=sys.executable)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--num-tests", type=int, default=10_000)
        subparser.add_argument("--seed", type=int, default=default_seed)
        subparser.add_argument(
            "--verify-existing-checkpoint",
            action="store_true",
            help="Verify the workspace checkpoint without first training a candidate.",
        )
        subparser.set_defaults(handler=evaluate_adderboard, layer=layer)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

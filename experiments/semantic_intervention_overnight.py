#!/usr/bin/env python3
"""Detached, cooperatively controllable semantic-campaign launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c0c3_factorial.semantic_interventions import (  # noqa: E402
    semantic_status,
    set_semantic_control,
)


def _session(campaign: Path) -> str:
    digest = hashlib.sha256(str(campaign.resolve()).encode()).hexdigest()[:10]
    return f"rl4rl-semantic-{digest}"


def _python_bin(path: Path) -> str:
    """Return an absolute interpreter path without resolving venv symlinks."""

    return str(path.absolute())


def _screen_running(session: str) -> bool:
    result = subprocess.run(
        ("screen", "-ls", session), capture_output=True, text=True, check=False
    )
    return result.returncode == 0 and session in result.stdout


def _launch(args: argparse.Namespace) -> dict[str, object]:
    campaign = args.campaign.resolve()
    session = _session(campaign)
    if _screen_running(session):
        return {"status": "already-running", "screen_session": session}
    if shutil.which("screen") is None:
        raise RuntimeError("screen is required for detached semantic campaigns")
    set_semantic_control(campaign, desired="running", reason=args.reason)
    log = campaign / "semantic-supervisor.log"
    command = [
        _python_bin(args.python_bin),
        str(
            args.runtime_root.resolve()
            / "experiments/semantic_intervention_experiment.py"
        ),
        "run-campaign",
        "--campaign",
        str(campaign),
        "--repo-root",
        str(args.runtime_root.resolve()),
        "--python-bin",
        _python_bin(args.python_bin),
        "--max-workers",
        str(args.max_workers),
        "--recover-interrupted",
    ]
    shell = (
        f"cd {shlex.quote(str(args.runtime_root.resolve()))} && "
        + " ".join(shlex.quote(value) for value in command)
        + f" >> {shlex.quote(str(log))} 2>&1"
    )
    subprocess.run(("screen", "-DmS", session, "zsh", "-lc", shell), check=True)
    return {
        "status": "started",
        "screen_session": session,
        "log": str(log),
        "max_workers": args.max_workers,
    }


def command_status(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    value = semantic_status(campaign)
    value["supervisor"] = {
        "screen_session": _session(campaign),
        "running": _screen_running(_session(campaign)),
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def command_control(args: argparse.Namespace) -> int:
    campaign = args.campaign.resolve()
    desired = "paused" if args.command == "pause" else "stopped"
    print(
        json.dumps(
            set_semantic_control(campaign, desired=desired, reason=args.reason),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "resume"):
        item = sub.add_parser(name)
        item.add_argument("--campaign", type=Path, required=True)
        item.add_argument("--runtime-root", type=Path, required=True)
        item.add_argument("--python-bin", type=Path, default=Path(sys.executable))
        item.add_argument("--max-workers", type=int, default=12)
        item.add_argument(
            "--reason", default=f"operator requested semantic campaign {name}"
        )
        item.set_defaults(
            handler=lambda args: (print(json.dumps(_launch(args), indent=2)), 0)[1]
        )
    status = sub.add_parser("status")
    status.add_argument("--campaign", type=Path, required=True)
    status.set_defaults(handler=command_status)
    for name in ("pause", "stop"):
        item = sub.add_parser(name)
        item.add_argument("--campaign", type=Path, required=True)
        item.add_argument("--reason", required=True)
        item.set_defaults(handler=command_control)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

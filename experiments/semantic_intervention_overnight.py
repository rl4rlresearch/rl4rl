#!/usr/bin/env python3
"""Detached, cooperatively controllable semantic-campaign launcher."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c0c3_factorial.fashion_mnist import DATA_ROOT_ENV  # noqa: E402
from experiments.c0c3_factorial.semantic_interventions import (  # noqa: E402
    semantic_status,
    set_semantic_control,
)

SEMANTIC_SUPERVISOR_METADATA = "semantic-supervisor.json"


def _session(campaign: Path) -> str:
    digest = hashlib.sha256(str(campaign.resolve()).encode()).hexdigest()[:10]
    return f"rl4rl-semantic-{digest}"


def _python_bin(path: Path) -> str:
    """Return an absolute interpreter path without resolving venv symlinks."""

    return str(path.absolute())


def _fashion_data_root(args: argparse.Namespace, campaign: Path) -> Path | None:
    configured = getattr(args, "fashion_data_root", None)
    if configured is not None:
        return Path(configured).expanduser().resolve()
    environment = os.environ.get(DATA_ROOT_ENV)
    if environment:
        return Path(environment).expanduser().resolve()
    # Official campaign paths live below <checkout>/data/c0c3. This fallback
    # keeps detached runtimes independent from their own empty data directory.
    if campaign.parent.name == "c0c3" and campaign.parent.parent.name == "data":
        inferred = campaign.parent.parent.parent / "data/raw/fashion-mnist"
        if inferred.is_dir():
            return inferred.resolve()
    return None


def _runtime_worker_count(campaign: Path, requested: int) -> int:
    """Preserve zero so a live-extensible runtime remains truly unbounded."""

    if requested < 0:
        raise ValueError("max_workers must be nonnegative")
    return requested


def _screen_running(session: str) -> bool:
    result = subprocess.run(
        (shutil.which("screen") or "/usr/bin/screen", "-ls"),
        capture_output=True,
        text=True,
        check=False,
    )
    for line in (result.stdout + result.stderr).splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        numbered_session = fields[0]
        _pid, separator, session_name = numbered_session.partition(".")
        if separator and session_name == session:
            return True
    return False


def _launch(args: argparse.Namespace) -> dict[str, object]:
    campaign = args.campaign.resolve()
    scientific_repo_root = (
        args.scientific_repo_root.resolve()
        if getattr(args, "scientific_repo_root", None) is not None
        else args.runtime_root.resolve()
    )
    runtime_worker_count = _runtime_worker_count(campaign, args.max_workers)
    session = _session(campaign)
    fashion_data_root = _fashion_data_root(args, campaign)
    metadata_path = campaign / SEMANTIC_SUPERVISOR_METADATA
    temporary = metadata_path.with_name(
        f".{metadata_path.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "campaign": str(campaign),
                "runtime_root": str(args.runtime_root.resolve()),
                "scientific_repo_root": str(scientific_repo_root),
                "python_bin": _python_bin(args.python_bin),
                "fashion_data_root": (
                    str(fashion_data_root) if fashion_data_root is not None else None
                ),
                "max_workers": args.max_workers,
                "screen_session": session,
                "updated_at": dt.datetime.now(dt.UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, metadata_path)
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
        str(scientific_repo_root),
        "--python-bin",
        _python_bin(args.python_bin),
        "--max-workers",
        str(runtime_worker_count),
        "--recover-interrupted",
    ]
    if fashion_data_root is not None:
        command.extend(("--fashion-data-root", str(fashion_data_root)))
    shell = (
        f"cd {shlex.quote(str(args.runtime_root.resolve()))} && "
        + " ".join(shlex.quote(value) for value in command)
        + f" >> {shlex.quote(str(log))} 2>&1"
    )
    result = subprocess.run(
        ("screen", "-dmS", session, "zsh", "-lc", shell), check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"screen failed to start (exit {result.returncode})")
    for _ in range(20):
        if _screen_running(session):
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(
            f"screen session {session!r} exited before reporting healthy"
        )
    return {
        "status": "started",
        "screen_session": session,
        "log": str(log),
        "max_workers": None if args.max_workers == 0 else args.max_workers,
        "effective_worker_count": (
            None if runtime_worker_count == 0 else runtime_worker_count
        ),
        "concurrency_policy": ("all_runnable" if args.max_workers == 0 else "bounded"),
    }


def command_resume_after_drain(args: argparse.Namespace) -> int:
    """Wait for graceful drains, then reload campaigns on the requested runtime."""

    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    campaigns = [path.resolve() for path in args.campaign]
    while any(_screen_running(_session(campaign)) for campaign in campaigns):
        time.sleep(args.poll_seconds)
    results: list[dict[str, object]] = []
    for campaign in campaigns:
        launch_args = argparse.Namespace(
            campaign=campaign,
            runtime_root=args.runtime_root,
            scientific_repo_root=args.scientific_repo_root,
            python_bin=args.python_bin,
            fashion_data_root=args.fashion_data_root,
            max_workers=args.max_workers,
            reason=args.reason,
        )
        results.append(_launch(launch_args))
    print(json.dumps({"status": "reloaded", "campaigns": results}, indent=2))
    return 0


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
        item.add_argument(
            "--scientific-repo-root",
            type=Path,
            help=(
                "optional frozen scientific source root; runtime code still loads "
                "from --runtime-root"
            ),
        )
        item.add_argument("--python-bin", type=Path, default=Path(sys.executable))
        item.add_argument("--fashion-data-root", type=Path)
        item.add_argument(
            "--max-workers",
            type=int,
            default=0,
            help="maximum simultaneous subject calls; 0 removes the limit",
        )
        item.add_argument(
            "--reason", default=f"operator requested semantic campaign {name}"
        )
        item.set_defaults(
            handler=lambda args: (print(json.dumps(_launch(args), indent=2)), 0)[1]
        )
    reload_after_drain = sub.add_parser("resume-after-drain")
    reload_after_drain.add_argument(
        "--campaign", type=Path, action="append", required=True
    )
    reload_after_drain.add_argument("--runtime-root", type=Path, required=True)
    reload_after_drain.add_argument("--scientific-repo-root", type=Path)
    reload_after_drain.add_argument(
        "--python-bin", type=Path, default=Path(sys.executable)
    )
    reload_after_drain.add_argument("--fashion-data-root", type=Path)
    reload_after_drain.add_argument("--max-workers", type=int, default=0)
    reload_after_drain.add_argument("--poll-seconds", type=float, default=5.0)
    reload_after_drain.add_argument(
        "--reason", default="graceful orchestrator runtime reload after drain"
    )
    reload_after_drain.set_defaults(handler=command_resume_after_drain)
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

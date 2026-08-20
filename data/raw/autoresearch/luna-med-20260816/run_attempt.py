#!/usr/bin/env python3
# ruff: noqa: E501
"""Record, evaluate, retain, and restore one local Autoresearch candidate."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ACCURACY_RE = re.compile(r"Results: (\d+)/(\d+) correct \(([0-9.]+)%\)")
PARAMETERS_RE = re.compile(r"Parameters \(unique\):\s*(\d+)")
AUTOMATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


class RunnerLockedError(RuntimeError):
    """Raised when another mutating runner command owns this run directory."""


@contextmanager
def _exclusive_run_lock(run_dir: Path) -> Iterator[None]:
    """Hold the run's advisory lock for one mutating runner invocation.

    ``flock`` is released automatically if the process exits or crashes, so a
    leftover lock file never blocks recovery.  The file's contents are only
    diagnostic; lock ownership is determined by the operating-system lock.
    """
    lock_path = run_dir.resolve() / ".runner.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            holder = handle.read().strip()
            suffix = f" Current holder metadata: {holder}" if holder else ""
            raise RunnerLockedError(
                "another runner command is already modifying this run; "
                "wait for it to finish before retrying." + suffix
            ) from error
        try:
            handle.seek(0)
            handle.truncate()
            json.dump(
                {
                    "pid": os.getpid(),
                    "started_at_utc": datetime.now(UTC).isoformat(),
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git(workspace: Path, *args: str, capture: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(workspace), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def _candidate_files(workspace: Path) -> tuple[Path, ...]:
    return (
        workspace / "src" / "model.py",
        workspace / "src" / "data.py",
        workspace / "src" / "train.py",
        workspace / "submission.py",
    )


def _assert_run_layout(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    required = (
        run_dir / "RUN_CONFIG.json",
        run_dir / "RUN_MANIFEST.json",
        run_dir / "STATE.json",
        run_dir / "RESULTS.tsv",
        run_dir / "official_verify.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    workspace = run_dir / "workspace"
    if missing or not (workspace / ".git").exists():
        raise ValueError(
            f"not an initialized Autoresearch run: {run_dir}; missing {missing}"
        )
    return workspace


def _safe_tsv(value: str) -> str:
    return " ".join(value.replace("\t", " ").replace("\n", " ").split())


def _append_result(run_dir: Path, row: dict[str, object]) -> None:
    fields = (
        "attempt_id",
        "commit",
        "parent_commit",
        "timestamp_utc",
        "accuracy",
        "parameters",
        "status",
        "description",
        "proposal",
        "train_exit",
        "verify_exit",
    )
    with (run_dir / "RESULTS.tsv").open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writerow({field: _safe_tsv(str(row.get(field, ""))) for field in fields})


def _append_micro_result(
    run_dir: Path, automation: dict[str, Any], event: dict[str, object]
) -> None:
    fields = (
        "macro_attempt_id",
        "automation_id",
        "micro_attempt_id",
        "commit",
        "parent_commit",
        "timestamp_utc",
        "family",
        "accuracy",
        "parameters",
        "status",
        "description",
        "proposal",
        "train_exit",
        "verify_exit",
    )
    row = {
        "macro_attempt_id": automation["attempt_id"],
        "automation_id": automation["automation_id"],
        "micro_attempt_id": event["attempt_id"],
        "family": automation["family"],
        **event,
    }
    with (run_dir / "AUTOMATION_RESULTS.tsv").open(
        "a", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writerow({field: _safe_tsv(str(row.get(field, ""))) for field in fields})


def _copy_candidate(workspace: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for source in _candidate_files(workspace):
        if not source.is_file():
            raise FileNotFoundError(f"candidate contract file missing: {source}")
        relative = source.relative_to(workspace)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _prepare_attempt_dir(run_dir: Path, attempt_dir: Path) -> None:
    """Create an attempt directory, preserving an interrupted partial attempt.

    This runs only after the exclusive runner lock was acquired, so an existing
    directory cannot belong to a live runner. A completed result is a state
    integrity conflict and must be repaired explicitly rather than overwritten.
    """
    if attempt_dir.exists():
        if (attempt_dir / "result.json").is_file():
            raise RuntimeError(
                "state conflicts with an already-recorded attempt directory: "
                f"{attempt_dir}. Repair the state before continuing."
            )
        archive_root = run_dir / "interrupted-attempts"
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        archived = archive_root / f"{attempt_dir.name}-{timestamp}"
        archive_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(attempt_dir), str(archived))
        print(f"Archived interrupted partial attempt to {archived}")
    attempt_dir.mkdir()


def _commit_candidate(
    workspace: Path, parent_commit: str, attempt_id: str, message: str
) -> str:
    _git(
        workspace,
        "add",
        *(str(path.relative_to(workspace)) for path in _candidate_files(workspace)),
        capture=False,
    )
    tree = _git(workspace, "write-tree")
    commit = _git(
        workspace,
        "commit-tree",
        tree,
        "-p",
        parent_commit,
        "-m",
        f"autoresearch {attempt_id}: {message}",
    )
    _git(
        workspace,
        "update-ref",
        f"refs/autoresearch/attempts/{attempt_id}",
        commit,
        capture=False,
    )
    return commit


def _run_command(
    command: list[str], *, cwd: Path, timeout: int, stdout: Path, stderr: Path
) -> int:
    try:
        with (
            stdout.open("w", encoding="utf-8") as out,
            stderr.open("w", encoding="utf-8") as err,
        ):
            completed = subprocess.run(
                command, cwd=cwd, stdout=out, stderr=err, timeout=timeout
            )
        return completed.returncode
    except subprocess.TimeoutExpired as error:
        stderr.write_text(
            f"Timed out after {timeout} seconds while running: {' '.join(command)}\n{error}\n",
            encoding="utf-8",
        )
        return 124


def _parse_verification(stdout: Path) -> dict[str, object]:
    text = (
        stdout.read_text(encoding="utf-8", errors="replace") if stdout.exists() else ""
    )
    result = ACCURACY_RE.search(text)
    parameters = PARAMETERS_RE.search(text)
    if result:
        correct, total, percent = result.groups()
        accuracy: float | None = float(percent) / 100
    else:
        correct = total = percent = None
        accuracy = None
    return {
        "correct": int(correct) if correct is not None else None,
        "total": int(total) if total is not None else None,
        "accuracy": accuracy,
        "accuracy_percent": float(percent) if percent is not None else None,
        "parameters": int(parameters.group(1)) if parameters else None,
        "qualified": "Status: QUALIFIED" in text,
    }


def _restore_incumbent(run_dir: Path, workspace: Path, state: dict[str, Any]) -> None:
    incumbent = state["incumbent"]
    _git(workspace, "reset", "--hard", incumbent["commit"], capture=False)
    checkpoint = run_dir / incumbent["checkpoint"]
    target = workspace / "checkpoints" / "best.pt"
    if checkpoint.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, target)


def _accept_incumbent(
    run_dir: Path,
    workspace: Path,
    state: dict[str, Any],
    commit: str,
    attempt_id: str,
    parameters: int,
    checkpoint: Path,
) -> None:
    incumbent_checkpoint = run_dir / "state" / "incumbent.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"candidate did not create checkpoint: {checkpoint}")
    shutil.copy2(checkpoint, incumbent_checkpoint)
    _git(workspace, "update-ref", "refs/heads/main", commit, capture=False)
    _git(workspace, "reset", "--hard", commit, capture=False)
    shutil.copy2(incumbent_checkpoint, checkpoint)
    state["incumbent"] = {
        "attempt_id": attempt_id,
        "commit": commit,
        "parameters": parameters,
        "checkpoint": "state/incumbent.pt",
    }


def _record_baseline(run_dir: Path) -> int:
    run_dir = run_dir.resolve()
    workspace = _assert_run_layout(run_dir)
    state = _read_json(run_dir / "STATE.json")
    config = _read_json(run_dir / "RUN_CONFIG.json")
    attempt_dir = run_dir / "attempts" / "0000-baseline"
    repairing_artifacts = (
        state["baseline_recorded"] and not (attempt_dir / "result.json").is_file()
    )
    if state["baseline_recorded"] and not repairing_artifacts:
        print("Baseline already recorded; refusing to duplicate it.")
        return 0

    attempt_dir.mkdir(exist_ok=True)
    _copy_candidate(workspace, attempt_dir / "candidate")
    verify_exit = _run_command(
        [
            config["python_bin"],
            str(run_dir / "official_verify.py"),
            str(workspace / "submission.py"),
        ],
        cwd=run_dir,
        timeout=int(config["verification_timeout_seconds"]),
        stdout=attempt_dir / "verify.stdout.log",
        stderr=attempt_dir / "verify.stderr.log",
    )
    verification = _parse_verification(attempt_dir / "verify.stdout.log")
    checkpoint = workspace / "checkpoints" / "best.pt"
    if checkpoint.is_file():
        shutil.copy2(checkpoint, attempt_dir / "checkpoint.pt")

    parameters = verification["parameters"]
    qualified = bool(verification["qualified"] and verify_exit == 0)
    status = "keep" if qualified else "error"
    event = {
        "attempt_id": "baseline",
        "commit": state["incumbent"]["commit"],
        "parent_commit": "",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "accuracy": ""
        if verification["accuracy_percent"] is None
        else f"{verification['accuracy_percent']:.6f}%",
        "parameters": "" if parameters is None else parameters,
        "status": status,
        "description": "initial 6080-parameter pretrained starting model",
        "proposal": "baseline qualification check",
        "train_exit": "",
        "verify_exit": verify_exit,
    }
    _write_json(
        attempt_dir / "result.json", {"event": event, "verification": verification}
    )
    if not repairing_artifacts:
        _append_result(run_dir, event)
    if not qualified or not isinstance(parameters, int):
        print(
            "Baseline failed official qualification; see attempts/0000-baseline.",
            file=sys.stderr,
        )
        return 1

    if repairing_artifacts:
        expected_parameters = state["incumbent"]["parameters"]
        if parameters != expected_parameters:
            raise RuntimeError(
                "recreated baseline does not match recorded incumbent: "
                f"expected {expected_parameters}, got {parameters}"
            )
        print(
            f"Recreated missing baseline artifacts: {parameters} parameters, "
            f"{verification['accuracy_percent']:.4f}%"
        )
        return 0

    state["baseline_recorded"] = True
    state["incumbent"]["parameters"] = parameters
    shutil.copy2(checkpoint, run_dir / "state" / "incumbent.pt")
    _write_json(run_dir / "STATE.json", state)
    print(
        f"Baseline recorded: {parameters} parameters, {verification['accuracy_percent']:.4f}%"
    )
    return 0


def _record_attempt(run_dir: Path, description: str, proposal: str) -> int:
    run_dir = run_dir.resolve()
    workspace = _assert_run_layout(run_dir)
    state = _read_json(run_dir / "STATE.json")
    config = _read_json(run_dir / "RUN_CONFIG.json")
    if state.get("active_automation"):
        raise RuntimeError(
            "finish the active automation before recording a regular attempt"
        )
    if not state["baseline_recorded"]:
        raise RuntimeError(
            "record the baseline first: run_attempt.py --run-dir RUN baseline"
        )
    if state["attempts_used"] >= int(config["max_attempts"]):
        print("Attempt budget exhausted; no command was run.")
        return 2

    number = int(state["attempts_used"]) + 1
    attempt_id = f"attempt-{number:04d}"
    attempt_dir = run_dir / "attempts" / attempt_id
    _prepare_attempt_dir(run_dir, attempt_dir)
    parent_commit = str(state["incumbent"]["commit"])
    commit = _commit_candidate(workspace, parent_commit, attempt_id, description)
    _copy_candidate(workspace, attempt_dir / "candidate")

    checkpoint = workspace / "checkpoints" / "best.pt"
    # A failed training run must never be evaluated through the prior
    # incumbent's checkpoint. The prior artifact is already safe in state/.
    checkpoint.unlink(missing_ok=True)
    train_exit = _run_command(
        list(config["training_command"]),
        cwd=workspace,
        timeout=int(config["training_timeout_seconds"]),
        stdout=attempt_dir / "train.stdout.log",
        stderr=attempt_dir / "train.stderr.log",
    )
    verify_exit = ""
    verification: dict[str, object] = {
        "correct": None,
        "total": None,
        "accuracy": None,
        "accuracy_percent": None,
        "parameters": None,
        "qualified": False,
    }
    if train_exit == 0 and checkpoint.is_file():
        verify_exit = _run_command(
            [
                config["python_bin"],
                str(run_dir / "official_verify.py"),
                str(workspace / "submission.py"),
            ],
            cwd=run_dir,
            timeout=int(config["verification_timeout_seconds"]),
            stdout=attempt_dir / "verify.stdout.log",
            stderr=attempt_dir / "verify.stderr.log",
        )
        verification = _parse_verification(attempt_dir / "verify.stdout.log")
    else:
        (attempt_dir / "verify.stdout.log").write_text(
            "Verification was not run.\n", encoding="utf-8"
        )
        (attempt_dir / "verify.stderr.log").write_text(
            "Training failed or produced no checkpoint.\n", encoding="utf-8"
        )

    if train_exit == 0 and checkpoint.is_file():
        shutil.copy2(checkpoint, attempt_dir / "checkpoint.pt")
    parameters = verification["parameters"]
    qualified = bool(verification["qualified"] and verify_exit == 0)
    incumbent_parameters = state["incumbent"]["parameters"]
    accepted = (
        qualified and isinstance(parameters, int) and parameters < incumbent_parameters
    )
    status = (
        "keep"
        if accepted
        else ("discard" if train_exit == 0 and verify_exit == 0 else "error")
    )
    event = {
        "attempt_id": attempt_id,
        "commit": commit,
        "parent_commit": parent_commit,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "accuracy": ""
        if verification["accuracy_percent"] is None
        else f"{verification['accuracy_percent']:.6f}%",
        "parameters": "" if parameters is None else parameters,
        "status": status,
        "description": description,
        "proposal": proposal,
        "train_exit": train_exit,
        "verify_exit": verify_exit,
    }
    _write_json(
        attempt_dir / "result.json",
        {"event": event, "verification": verification, "accepted": accepted},
    )
    _append_result(run_dir, event)
    state["attempts_used"] = number
    if accepted:
        _accept_incumbent(
            run_dir, workspace, state, commit, attempt_id, parameters, checkpoint
        )
    else:
        _restore_incumbent(run_dir, workspace, state)
    _write_json(run_dir / "STATE.json", state)

    decision = "ACCEPTED" if accepted else "DISCARDED"
    accuracy = verification["accuracy_percent"]
    print(f"{attempt_id} {decision}: parameters={parameters}, accuracy={accuracy}")
    return 0


def _start_automation(
    run_dir: Path,
    automation_id: str,
    family: str,
    description: str,
    proposal: str,
    max_micro_trials: int,
) -> int:
    run_dir = run_dir.resolve()
    _assert_run_layout(run_dir)
    state = _read_json(run_dir / "STATE.json")
    config = _read_json(run_dir / "RUN_CONFIG.json")
    if not AUTOMATION_ID_RE.fullmatch(automation_id):
        raise ValueError(
            "automation id must be 3-64 lowercase letters, digits, and hyphens"
        )
    if max_micro_trials < 1:
        raise ValueError("--max-micro-trials must be at least 1")
    if not state["baseline_recorded"]:
        raise RuntimeError("record the baseline before starting an automation")
    if state.get("active_automation"):
        raise RuntimeError("an automation is already active")
    if state["attempts_used"] >= int(config["max_attempts"]):
        print("Attempt budget exhausted; no automation was started.")
        return 2

    number = int(state["attempts_used"]) + 1
    attempt_id = f"attempt-{number:04d}"
    automation_dir = run_dir / "automations" / attempt_id
    automation_dir.mkdir(parents=True)
    automation = {
        "schema": "rl4rl-autoresearch-automation-v1",
        "attempt_id": attempt_id,
        "automation_id": automation_id,
        "family": family,
        "description": description,
        "proposal": proposal,
        "max_micro_trials": max_micro_trials,
        "micro_attempts_used": 0,
        "accepted_micro_trials": 0,
        "error_micro_trials": 0,
        "parent_commit": state["incumbent"]["commit"],
        "parent_parameters": state["incumbent"]["parameters"],
        "best_accuracy": "",
        "best_parameters": "",
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    state["attempts_used"] = number
    state["active_automation"] = automation
    _write_json(automation_dir / "AUTOMATION.json", automation)
    _write_json(run_dir / "STATE.json", state)
    print(
        f"Started {attempt_id} ({automation_id}); micro-trial cap: {max_micro_trials}"
    )
    return 0


def _record_automation_attempt(run_dir: Path, description: str, proposal: str) -> int:
    run_dir = run_dir.resolve()
    workspace = _assert_run_layout(run_dir)
    state = _read_json(run_dir / "STATE.json")
    config = _read_json(run_dir / "RUN_CONFIG.json")
    automation = state.get("active_automation")
    if not automation:
        raise RuntimeError("start an automation before recording a micro-trial")
    if int(automation["micro_attempts_used"]) >= int(automation["max_micro_trials"]):
        print("Micro-trial budget exhausted; no command was run.")
        return 2

    number = int(automation["micro_attempts_used"]) + 1
    micro_id = f"micro-{number:04d}"
    attempt_dir = run_dir / "automations" / str(automation["attempt_id"]) / micro_id
    _prepare_attempt_dir(run_dir, attempt_dir)
    parent_commit = str(state["incumbent"]["commit"])
    composite_id = f"{automation['attempt_id']}/{micro_id}"
    commit = _commit_candidate(workspace, parent_commit, composite_id, description)
    _copy_candidate(workspace, attempt_dir / "candidate")

    checkpoint = workspace / "checkpoints" / "best.pt"
    checkpoint.unlink(missing_ok=True)
    train_exit = _run_command(
        list(config["training_command"]),
        cwd=workspace,
        timeout=int(config["training_timeout_seconds"]),
        stdout=attempt_dir / "train.stdout.log",
        stderr=attempt_dir / "train.stderr.log",
    )
    verify_exit = ""
    verification: dict[str, object] = {
        "correct": None,
        "total": None,
        "accuracy": None,
        "accuracy_percent": None,
        "parameters": None,
        "qualified": False,
    }
    if train_exit == 0 and checkpoint.is_file():
        verify_exit = _run_command(
            [
                config["python_bin"],
                str(run_dir / "official_verify.py"),
                str(workspace / "submission.py"),
            ],
            cwd=run_dir,
            timeout=int(config["verification_timeout_seconds"]),
            stdout=attempt_dir / "verify.stdout.log",
            stderr=attempt_dir / "verify.stderr.log",
        )
        verification = _parse_verification(attempt_dir / "verify.stdout.log")
    else:
        (attempt_dir / "verify.stdout.log").write_text(
            "Verification was not run.\n", encoding="utf-8"
        )
        (attempt_dir / "verify.stderr.log").write_text(
            "Training failed or produced no checkpoint.\n", encoding="utf-8"
        )

    if train_exit == 0 and checkpoint.is_file():
        shutil.copy2(checkpoint, attempt_dir / "checkpoint.pt")
    parameters = verification["parameters"]
    qualified = bool(verification["qualified"] and verify_exit == 0)
    accepted = (
        qualified
        and isinstance(parameters, int)
        and parameters < state["incumbent"]["parameters"]
    )
    status = (
        "keep"
        if accepted
        else ("discard" if train_exit == 0 and verify_exit == 0 else "error")
    )
    event = {
        "attempt_id": micro_id,
        "commit": commit,
        "parent_commit": parent_commit,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "accuracy": ""
        if verification["accuracy_percent"] is None
        else f"{verification['accuracy_percent']:.6f}%",
        "parameters": "" if parameters is None else parameters,
        "status": status,
        "description": description,
        "proposal": proposal,
        "train_exit": train_exit,
        "verify_exit": verify_exit,
    }
    _write_json(
        attempt_dir / "result.json",
        {"event": event, "verification": verification, "accepted": accepted},
    )
    _append_micro_result(run_dir, automation, event)
    automation["micro_attempts_used"] = number
    if accepted:
        _accept_incumbent(
            run_dir, workspace, state, commit, composite_id, parameters, checkpoint
        )
        automation["accepted_micro_trials"] = (
            int(automation["accepted_micro_trials"]) + 1
        )
        automation["best_accuracy"] = event["accuracy"]
        automation["best_parameters"] = parameters
    else:
        if status == "error":
            automation["error_micro_trials"] = int(automation["error_micro_trials"]) + 1
        _restore_incumbent(run_dir, workspace, state)
    state["active_automation"] = automation
    _write_json(
        run_dir / "automations" / str(automation["attempt_id"]) / "AUTOMATION.json",
        automation,
    )
    _write_json(run_dir / "STATE.json", state)

    decision = "ACCEPTED" if accepted else "DISCARDED"
    print(f"{automation['attempt_id']} {micro_id} {decision}: parameters={parameters}")
    return 0


def _finish_automation(run_dir: Path, summary: str) -> int:
    run_dir = run_dir.resolve()
    _assert_run_layout(run_dir)
    state = _read_json(run_dir / "STATE.json")
    automation = state.get("active_automation")
    if not automation:
        raise RuntimeError("there is no active automation to finish")
    incumbent = state["incumbent"]
    improved = incumbent["parameters"] < automation["parent_parameters"]
    status = (
        "keep"
        if improved
        else ("error" if automation["micro_attempts_used"] == 0 else "discard")
    )
    event = {
        "attempt_id": automation["attempt_id"],
        "commit": incumbent["commit"] if improved else automation["parent_commit"],
        "parent_commit": automation["parent_commit"],
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "accuracy": automation["best_accuracy"] if improved else "",
        "parameters": incumbent["parameters"] if improved else "",
        "status": status,
        "description": f"[automation:{automation['automation_id']}] {automation['description']} | {summary}",
        "proposal": automation["proposal"],
        "train_exit": "",
        "verify_exit": "",
    }
    _append_result(run_dir, event)
    automation["finished_at_utc"] = datetime.now(UTC).isoformat()
    automation["status"] = status
    automation["summary"] = summary
    _write_json(
        run_dir / "automations" / str(automation["attempt_id"]) / "AUTOMATION.json",
        automation,
    )
    state.pop("active_automation")
    _write_json(run_dir / "STATE.json", state)
    print(f"Finished {event['attempt_id']} ({automation['automation_id']}): {status}")
    return 0


def _show_status(run_dir: Path) -> int:
    run_dir = run_dir.resolve()
    _assert_run_layout(run_dir)
    state = _read_json(run_dir / "STATE.json")
    config = _read_json(run_dir / "RUN_CONFIG.json")
    incumbent = state["incumbent"]
    print(f"baseline_recorded: {state['baseline_recorded']}")
    print(f"attempts: {state['attempts_used']}/{config['max_attempts']}")
    print(
        f"incumbent: {incumbent['attempt_id']} ({incumbent['parameters']} parameters)"
    )
    print(f"commit: {incumbent['commit']}")
    if automation := state.get("active_automation"):
        print(
            "active_automation: "
            f"{automation['attempt_id']} ({automation['automation_id']}, "
            f"{automation['micro_attempts_used']}/{automation['max_micro_trials']} micro-trials)"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "baseline", help="verify and freeze the initial 6080p candidate"
    )
    attempt = subcommands.add_parser("attempt", help="train and log one candidate")
    attempt.add_argument("--description", required=True)
    attempt.add_argument("--proposal", required=True)
    automation_start = subcommands.add_parser(
        "automation-start", help="reserve one macro attempt and begin an automation"
    )
    automation_start.add_argument("--automation-id", required=True)
    automation_start.add_argument("--family", required=True)
    automation_start.add_argument("--description", required=True)
    automation_start.add_argument("--proposal", required=True)
    automation_start.add_argument("--max-micro-trials", required=True, type=int)
    automation_attempt = subcommands.add_parser(
        "automation-attempt",
        help="train and log one candidate in the active automation",
    )
    automation_attempt.add_argument("--description", required=True)
    automation_attempt.add_argument("--proposal", required=True)
    automation_end = subcommands.add_parser(
        "automation-end", help="write the active automation's one macro summary row"
    )
    automation_end.add_argument("--summary", required=True)
    subcommands.add_parser("status", help="show budget and retained incumbent")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "status":
        return _show_status(args.run_dir)
    try:
        with _exclusive_run_lock(args.run_dir):
            if args.command == "baseline":
                return _record_baseline(args.run_dir)
            if args.command == "attempt":
                return _record_attempt(args.run_dir, args.description, args.proposal)
            if args.command == "automation-start":
                return _start_automation(
                    args.run_dir,
                    args.automation_id,
                    args.family,
                    args.description,
                    args.proposal,
                    args.max_micro_trials,
                )
            if args.command == "automation-attempt":
                return _record_automation_attempt(
                    args.run_dir, args.description, args.proposal
                )
            return _finish_automation(args.run_dir, args.summary)
    except RunnerLockedError as error:
        print(f"Runner busy: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

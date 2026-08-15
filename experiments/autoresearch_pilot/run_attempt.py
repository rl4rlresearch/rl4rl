#!/usr/bin/env python3
# ruff: noqa: E501
"""Record, evaluate, retain, and restore one local Autoresearch candidate."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ACCURACY_RE = re.compile(r"Results: (\d+)/(\d+) correct \(([0-9.]+)%\)")
PARAMETERS_RE = re.compile(r"Parameters \(unique\):\s*(\d+)")


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


def _copy_candidate(workspace: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for source in _candidate_files(workspace):
        if not source.is_file():
            raise FileNotFoundError(f"candidate contract file missing: {source}")
        relative = source.relative_to(workspace)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


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
    attempt_dir.mkdir()
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
    subcommands.add_parser("status", help="show budget and retained incumbent")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "baseline":
        return _record_baseline(args.run_dir)
    if args.command == "attempt":
        return _record_attempt(args.run_dir, args.description, args.proposal)
    return _show_status(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())

"""Operational Codex transport guard for long-running campaign runtimes.

The scientific runtimes intentionally remain hash-pinned.  This module is
loaded by the host supervisor and wraps their Codex transport at process
startup.  A nonzero provider exit must not become a scientific proposal: the
guard archives the failed transport artifacts, restores the exact pre-call
workspace, backs off, and retries the same opportunity.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
import time
from contextlib import suppress
from functools import wraps
from pathlib import Path
from typing import Any

INITIAL_BACKOFF_ENV = "RL4RL_CODEX_PROVIDER_INITIAL_BACKOFF_SECONDS"
MAX_BACKOFF_ENV = "RL4RL_CODEX_PROVIDER_MAX_BACKOFF_SECONDS"
DEFAULT_INITIAL_BACKOFF_SECONDS = 30.0
DEFAULT_MAX_BACKOFF_SECONDS = 15 * 60.0


def _seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    value = default if raw is None else float(raw)
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        with suppress(FileNotFoundError):
            path.chmod(path.stat().st_mode | 0o700)


def _restore_workspace(workspace: Path, snapshot: Path) -> None:
    _make_tree_writable(workspace)
    shutil.rmtree(workspace)
    shutil.copytree(snapshot, workspace, symlinks=True)


def _error_summary(events_path: Path, stderr_path: Path) -> list[str]:
    messages: list[str] = []
    if events_path.is_file():
        for raw in events_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "error" and isinstance(
                event.get("message"), str
            ):
                messages.append(str(event["message"]))
            failed = event.get("error")
            if event.get("type") == "turn.failed" and isinstance(failed, dict):
                message = failed.get("message")
                if isinstance(message, str):
                    messages.append(message)
    if stderr_path.is_file():
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        if stderr:
            messages.append(stderr)
    # Enough context for diagnosis without duplicating a reconnect storm.
    return messages[-8:]


def _archive_failed_call(
    *,
    result: Any,
    log_root: Path,
    call_id: str,
    retry_number: int,
    delay_seconds: float,
) -> Path:
    retry_root = log_root / "provider-retries"
    destination = retry_root / f"{call_id}-retry-{retry_number:04d}"
    destination.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(result.events_path),
        Path(result.stderr_path),
        log_root / f"{call_id}.last-message.md",
    ]
    archived: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        target = destination / path.name
        shutil.move(str(path), str(target))
        archived.append(target.name)
    metadata = {
        "schema_version": "1.0",
        "event": "provider_transport_retry",
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "call_id": call_id,
        "retry_number": retry_number,
        "returncode": int(result.returncode),
        "delay_seconds": delay_seconds,
        "archived_files": archived,
        "errors": _error_summary(
            destination / Path(result.events_path).name,
            destination / Path(result.stderr_path).name,
        ),
    }
    (destination / "retry.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _jitter_seconds(call_id: str, retry_number: int, backoff: float) -> float:
    if backoff <= 0:
        return 0.0
    digest = hashlib.sha256(f"{call_id}:{retry_number}".encode()).digest()
    fraction = int.from_bytes(digest[:4], "big") / float(2**32 - 1)
    return fraction * min(10.0, backoff * 0.25)


def install_provider_retry_guard(codex_cli_module: Any) -> None:
    """Install the retry wrapper on one imported, possibly frozen runtime."""

    codex_class = codex_cli_module.CodexCli
    if getattr(codex_class.run, "_rl4rl_provider_guard", False):
        return
    original_run = codex_class.run

    @wraps(original_run)
    def guarded_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        workspace = Path(kwargs["workspace"]).resolve()
        log_root = Path(kwargs["log_root"]).resolve()
        call_id = str(kwargs["call_id"])
        initial = _seconds(INITIAL_BACKOFF_ENV, DEFAULT_INITIAL_BACKOFF_SECONDS)
        maximum = _seconds(MAX_BACKOFF_ENV, DEFAULT_MAX_BACKOFF_SECONDS)
        if maximum < initial:
            maximum = initial

        with tempfile.TemporaryDirectory(
            prefix="rl4rl-provider-workspace-snapshot-"
        ) as temporary:
            snapshot = Path(temporary) / "workspace"
            shutil.copytree(workspace, snapshot, symlinks=True)
            retry_number = 0
            backoff = initial
            while True:
                result = original_run(self, *args, **kwargs)
                # Exit 124 is the campaign's explicit subject-call timeout.  All
                # other nonzero exits are provider/infrastructure failures in
                # the frozen runner and must not spend a research opportunity.
                if int(result.returncode) in {0, 124}:
                    return result
                retry_number += 1
                delay = min(maximum, backoff) + _jitter_seconds(
                    call_id, retry_number, backoff
                )
                _archive_failed_call(
                    result=result,
                    log_root=log_root,
                    call_id=call_id,
                    retry_number=retry_number,
                    delay_seconds=delay,
                )
                _restore_workspace(workspace, snapshot)
                time.sleep(delay)
                backoff = min(maximum, max(initial, backoff * 2.0))

    guarded_run._rl4rl_provider_guard = True  # type: ignore[attr-defined]
    codex_class.run = guarded_run

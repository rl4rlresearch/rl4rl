"""Non-interactive Codex CLI transport with complete JSONL accounting."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .spec import ModelSpec
from .state import Usage


@dataclass(frozen=True)
class CodexResult:
    returncode: int
    last_message: str
    usage: Usage
    events_path: Path
    stderr_path: Path


def usage_from_events(path: Path) -> Usage:
    completed: dict[str, int] | None = None
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.completed" and isinstance(
                event.get("usage"), dict
            ):
                completed = event["usage"]
    if completed is None:
        return Usage()
    return Usage(
        input_tokens=int(completed.get("input_tokens", 0)),
        cached_input_tokens=int(completed.get("cached_input_tokens", 0)),
        output_tokens=int(completed.get("output_tokens", 0)),
        reasoning_output_tokens=int(completed.get("reasoning_output_tokens", 0)),
    )


class CodexCli:
    def __init__(self, binary: str = "codex") -> None:
        self.binary = binary

    def run(
        self,
        *,
        prompt: str,
        workspace: Path,
        model: ModelSpec,
        log_root: Path,
        call_id: str,
        sandbox: str | None = None,
        timeout_seconds: int = 3600,
    ) -> CodexResult:
        log_root.mkdir(parents=True, exist_ok=True)
        events = log_root / f"{call_id}.jsonl"
        stderr = log_root / f"{call_id}.stderr.log"
        last_message = log_root / f"{call_id}.last-message.md"
        if any(path.exists() for path in (events, stderr, last_message)):
            raise FileExistsError(f"Codex call ID already exists: {call_id}")
        command = [
            self.binary,
            "exec",
            "--model",
            model.name,
            "-c",
            f'model_reasoning_effort="{model.reasoning_effort}"',
            "--json",
            "--output-last-message",
            str(last_message),
            "--sandbox",
            sandbox or model.sandbox,
            "-a",
            model.approval_policy,
            "--ephemeral",
            "--skip-git-repo-check",
            "--cd",
            str(workspace),
            "-",
        ]
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{call_id}.", dir=log_root
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with (
                temporary.open("w", encoding="utf-8") as stdout_handle,
                stderr.open("w", encoding="utf-8") as stderr_handle,
            ):
                try:
                    completed = subprocess.run(
                        command,
                        input=prompt,
                        text=True,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        timeout=timeout_seconds,
                        check=False,
                    )
                    returncode = completed.returncode
                except subprocess.TimeoutExpired as error:
                    stderr_handle.write(f"\nCodex timeout: {error}\n")
                    returncode = 124
            temporary.replace(events)
        finally:
            temporary.unlink(missing_ok=True)
        message = (
            last_message.read_text(encoding="utf-8", errors="replace")
            if last_message.is_file()
            else ""
        )
        return CodexResult(
            returncode=returncode,
            last_message=message,
            usage=usage_from_events(events),
            events_path=events,
            stderr_path=stderr,
        )

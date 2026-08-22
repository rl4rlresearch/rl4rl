"""Non-interactive Codex CLI transport with complete JSONL accounting."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .environment import (
    controlled_subprocess_environment,
    subject_subprocess_environment,
)
from .spec import ModelSpec
from .state import Usage


@dataclass(frozen=True)
class CodexResult:
    returncode: int
    last_message: str
    usage: Usage
    events_path: Path
    stderr_path: Path
    session_id: str | None


def session_id_from_events(path: Path) -> str | None:
    """Return the Codex thread ID recorded for a non-ephemeral session."""

    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "thread.started":
            continue
        session_id = event.get("thread_id")
        if isinstance(session_id, str) and session_id.strip():
            return session_id
    return None


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
        run_seed: int | None = None,
        timeout_seconds: int = 3600,
        resume_session_id: str | None = None,
        persist_session: bool = False,
        neutral_subject: bool = False,
    ) -> CodexResult:
        log_root.mkdir(parents=True, exist_ok=True)
        events = log_root / f"{call_id}.jsonl"
        stderr = log_root / f"{call_id}.stderr.log"
        last_message = log_root / f"{call_id}.last-message.md"
        if any(path.exists() for path in (events, stderr, last_message)):
            raise FileExistsError(f"Codex call ID already exists: {call_id}")
        if resume_session_id is None:
            command = [
                self.binary,
                "exec",
                "--model",
                model.name,
                "-c",
                f'model_reasoning_effort="{model.reasoning_effort}"',
                "-c",
                f'approval_policy="{model.approval_policy}"',
                "--json",
                "--output-last-message",
                str(last_message),
                "--sandbox",
                sandbox or model.sandbox,
                "--skip-git-repo-check",
                "--cd",
                str(workspace),
            ]
            # A continuous session must be persisted by Codex so a later
            # opportunity can use ``codex exec resume``. Existing protocols
            # remain ephemeral and therefore behaviorally unchanged.
            if not persist_session:
                command.append("--ephemeral")
            command.append("-")
        else:
            command = [
                self.binary,
                "exec",
                "resume",
                "--model",
                model.name,
                "-c",
                f'model_reasoning_effort="{model.reasoning_effort}"',
                "-c",
                f'approval_policy="{model.approval_policy}"',
                "--json",
                "--output-last-message",
                str(last_message),
                "--skip-git-repo-check",
                resume_session_id,
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
                        env=(
                            subject_subprocess_environment(
                                run_seed, workspace=workspace
                            )
                            if neutral_subject
                            else controlled_subprocess_environment(run_seed)
                        ),
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
            session_id=session_id_from_events(events) or resume_session_id,
        )

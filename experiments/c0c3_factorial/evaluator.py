"""Framework-neutral Layer A command evaluator."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .artifacts import materialize_candidate
from .environment import controlled_subprocess_environment
from .spec import ObjectiveDirection, TaskSpec
from .state import Evaluation


@dataclass(frozen=True)
class EvaluationArtifacts:
    evaluation: Evaluation
    stdout_path: Path
    stderr_path: Path
    workspace_path: Path


def _format_command(
    command: tuple[str, ...],
    *,
    python_bin: str,
    workspace: Path,
    repo_root: Path,
    output: Path,
) -> list[str]:
    values = {
        "python": python_bin,
        "workspace": str(workspace),
        "repo_root": str(repo_root),
        "output": str(output),
    }
    return [token.format(**values) for token in command]


def parse_metrics(text: str, task: TaskSpec) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, pattern in task.metric_patterns.items():
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if not matches:
            continue
        value = matches[-1]
        if isinstance(value, tuple):
            value = value[-1]
        metrics[name] = float(str(value).replace(",", ""))
    return metrics


class CommandEvaluator:
    def __init__(
        self,
        *,
        task: TaskSpec,
        support_source: Path,
        repo_root: Path,
        python_bin: str,
    ) -> None:
        self.task = task
        self.support_source = support_source
        self.repo_root = repo_root
        if "/" in python_bin:
            # Make relative paths safe for the evaluator's temporary cwd without
            # dereferencing a virtual-environment interpreter symlink. Resolving
            # that symlink can bypass pyvenv.cfg and silently drop dependencies.
            self.python_bin = str(Path(python_bin).expanduser().absolute())
        else:
            self.python_bin = shutil.which(python_bin) or python_bin

    def evaluate(
        self,
        *,
        candidate_snapshot: Path,
        opportunity_root: Path,
        timeout_seconds: int,
        run_seed: int | None = None,
        verify_existing_checkpoint: bool = False,
    ) -> EvaluationArtifacts:
        workspace = opportunity_root / "evaluation-workspace"
        materialize_candidate(
            self.support_source,
            candidate_snapshot,
            workspace,
            self.task.editable_paths,
        )
        # Candidate proposals always train from scratch.  Calibration is the one
        # exception: it verifies the immutable task seed's supplied checkpoint.
        if not verify_existing_checkpoint:
            shutil.rmtree(workspace / "checkpoints", ignore_errors=True)
        output_json = opportunity_root / "evaluation.json"
        stdout = opportunity_root / "evaluation.stdout.log"
        stderr = opportunity_root / "evaluation.stderr.log"
        command = _format_command(
            self.task.evaluator_command,
            python_bin=self.python_bin,
            workspace=workspace,
            repo_root=self.repo_root,
            output=output_json,
        )
        if verify_existing_checkpoint:
            command.append("--verify-existing-checkpoint")
        started = time.monotonic()
        try:
            with (
                stdout.open("w", encoding="utf-8") as stdout_handle,
                stderr.open("w", encoding="utf-8") as stderr_handle,
            ):
                completed = subprocess.run(
                    command,
                    cwd=workspace,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=controlled_subprocess_environment(run_seed),
                    timeout=timeout_seconds,
                    check=False,
                )
            returncode = completed.returncode
            failure_kind = "execution" if returncode else None
        except subprocess.TimeoutExpired as error:
            stderr.write_text(f"Evaluator timeout: {error}\n", encoding="utf-8")
            returncode = 124
            failure_kind = "timeout"
        elapsed = time.monotonic() - started
        combined = "\n".join(
            (
                stdout.read_text(encoding="utf-8", errors="replace"),
                stderr.read_text(encoding="utf-8", errors="replace"),
            )
        )
        if returncode and "MODEL_CONTRACT_VIOLATION:" in combined:
            failure_kind = "model_contract"
        metrics: dict[str, float | int | str | bool | None]
        if output_json.is_file():
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            metrics = dict(payload.get("metrics", payload))
        else:
            metrics = parse_metrics(combined, self.task)
        valid = returncode == 0 and self.task.objective_metric in metrics
        if valid and self.task.qualification_metric is not None:
            qualification = metrics.get(self.task.qualification_metric)
            valid = (
                isinstance(qualification, int | float)
                and not isinstance(qualification, bool)
                and qualification >= float(self.task.qualification_minimum)
            )
            if not valid:
                failure_kind = "nonqualification"
        objective = metrics.get(self.task.objective_metric)
        if (
            valid
            and isinstance(objective, int | float)
            and not isinstance(objective, bool)
        ):
            fitness = float(objective)
            if self.task.objective_direction is ObjectiveDirection.MINIMIZE:
                fitness = -fitness
        else:
            valid = False
            fitness = None
            failure_kind = failure_kind or "missing_metric"
        evaluation = Evaluation(
            valid=valid,
            fitness=fitness,
            metrics=metrics,
            evaluator_seconds=elapsed,
            failure_kind=None if valid else failure_kind,
        )
        return EvaluationArtifacts(evaluation, stdout, stderr, workspace)

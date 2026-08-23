"""Framework-neutral Layer A command evaluator."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .artifacts import materialize_candidate
from .environment import controlled_subprocess_environment
from .spec import ObjectiveDirection, TaskSpec
from .state import Evaluation

SHARED_LOCAL_EVALUATOR_CAPACITY = 6
SHARED_LOCAL_EVALUATOR_ROOT_ENV = "RL4RL_SHARED_LOCAL_EVALUATOR_ROOT"


def shared_local_evaluator_root() -> Path:
    """Return one host-wide lock root shared by detached campaign runtimes."""

    configured = os.environ.get(SHARED_LOCAL_EVALUATOR_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    stable_tmp = Path("/private/tmp")
    base = stable_tmp if stable_tmp.is_dir() else Path(tempfile.gettempdir())
    return base / f"rl4rl-c0c3-local-evaluators-{os.getuid()}-v1"


@dataclass(frozen=True)
class EvaluationArtifacts:
    evaluation: Evaluation
    stdout_path: Path
    stderr_path: Path
    workspace_path: Path


@dataclass
class _SlotLease:
    handle: TextIO
    path: Path
    index: int


def _release_slot(lease: _SlotLease | None) -> None:
    if lease is None:
        return
    fcntl.flock(lease.handle.fileno(), fcntl.LOCK_UN)
    lease.handle.close()


def _slot_first(opportunity_root: Path, *, scope: str, capacity: int) -> int:
    value = f"{scope}\0{opportunity_root}"
    return (
        int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")
        % capacity
    )


def _try_acquire_slot(
    *,
    root: Path,
    capacity: int,
    first: int,
    opportunity_root: Path,
    scope: str,
) -> _SlotLease | None:
    for offset in range(capacity):
        index = (first + offset) % capacity
        path = root / f"slot-{index:02d}.lock"
        handle = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "schema_version": "1.0",
                "pid": os.getpid(),
                "scope": scope,
                "opportunity_root": str(opportunity_root),
                "acquired_at_unix": time.time(),
            },
            handle,
        )
        handle.flush()
        return _SlotLease(handle=handle, path=path, index=index)
    return None


def _ensure_shared_scheduler(root: Path, capacity: int) -> None:
    """Create or verify the immutable host-wide scheduler configuration."""

    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "scheduler.json"
    lock_path = root / "scheduler-config.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if config_path.is_file():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            if payload.get("capacity") != capacity:
                raise RuntimeError(
                    "shared local evaluator scheduler capacity mismatch: "
                    f"expected {capacity}, found {payload.get('capacity')}"
                )
        else:
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "capacity": capacity,
                        "scheduler": "crash_releasing_host_file_locks_v1",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def shared_local_evaluator_status(
    root: Path | None = None,
    capacity: int = SHARED_LOCAL_EVALUATOR_CAPACITY,
) -> dict[str, object]:
    """Inspect host-wide slots without disturbing an active lease."""

    selected_root = (root or shared_local_evaluator_root()).resolve()
    _ensure_shared_scheduler(selected_root, capacity)
    slots: list[dict[str, object]] = []
    for index in range(capacity):
        path = selected_root / f"slot-{index:02d}.lock"
        handle = path.open("a+", encoding="utf-8")
        occupied = False
        holder: dict[str, object] = {}
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            occupied = True
            handle.seek(0)
            try:
                value = json.loads(handle.read() or "{}")
            except json.JSONDecodeError:
                value = {}
            if isinstance(value, dict):
                holder = value
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
        slots.append(
            {
                "slot": index,
                "occupied": occupied,
                "holder": holder if occupied else None,
            }
        )
    return {
        "schema_version": "1.0",
        "root": str(selected_root),
        "capacity": capacity,
        "occupied": sum(bool(slot["occupied"]) for slot in slots),
        "available": sum(not bool(slot["occupied"]) for slot in slots),
        "slots": slots,
    }


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
        slot_root: Path | None = None,
        max_parallel_evaluators: int | None = None,
        shared_slot_root: Path | None = None,
        max_shared_parallel_evaluators: int | None = None,
    ) -> None:
        self.task = task
        self.support_source = support_source
        self.repo_root = repo_root
        if (slot_root is None) != (max_parallel_evaluators is None):
            raise ValueError(
                "slot_root and max_parallel_evaluators must be configured together"
            )
        if max_parallel_evaluators is not None and max_parallel_evaluators < 1:
            raise ValueError("max_parallel_evaluators must be positive")
        self.slot_root = slot_root
        self.max_parallel_evaluators = max_parallel_evaluators
        if (shared_slot_root is None) != (max_shared_parallel_evaluators is None):
            raise ValueError(
                "shared_slot_root and max_shared_parallel_evaluators must be "
                "configured together"
            )
        if max_shared_parallel_evaluators is not None and (
            max_shared_parallel_evaluators < 1
        ):
            raise ValueError("max_shared_parallel_evaluators must be positive")
        # Any campaign that already has a local per-campaign pool also joins the
        # host-wide pool. Keeping both leases preserves the frozen three-slot
        # campaign cap while preventing separate campaigns from oversubscribing
        # one Mac. Calibrations and serial post-search evaluators have no
        # campaign pool and therefore do not silently enter this scheduler.
        if slot_root is not None and shared_slot_root is None:
            shared_slot_root = shared_local_evaluator_root()
            max_shared_parallel_evaluators = SHARED_LOCAL_EVALUATOR_CAPACITY
        self.shared_slot_root = shared_slot_root
        self.max_shared_parallel_evaluators = max_shared_parallel_evaluators
        if "/" in python_bin:
            # Make relative paths safe for the evaluator's temporary cwd without
            # dereferencing a virtual-environment interpreter symlink. Resolving
            # that symlink can bypass pyvenv.cfg and silently drop dependencies.
            self.python_bin = str(Path(python_bin).expanduser().absolute())
        else:
            self.python_bin = shutil.which(python_bin) or python_bin

    @contextlib.contextmanager
    def _evaluation_slot(
        self, opportunity_root: Path, *, include_shared_local_pool: bool = True
    ):
        """Limit concurrent trainers within a campaign and across the host.

        File locks are released automatically if a worker crashes. Waiting is
        outside the evaluator timeout and budget so machine contention cannot
        turn a scientifically valid candidate into an artificial timeout.
        """

        if self.slot_root is None or self.max_parallel_evaluators is None:
            yield
            return

        campaign_root = self.slot_root.resolve()
        campaign_root.mkdir(parents=True, exist_ok=True)
        shared_root = (
            self.shared_slot_root.resolve()
            if include_shared_local_pool and self.shared_slot_root is not None
            else None
        )
        shared_capacity = (
            self.max_shared_parallel_evaluators
            if shared_root is not None
            else None
        )
        if shared_root is not None and shared_capacity is not None:
            if shared_root == campaign_root:
                raise ValueError("campaign and shared evaluator slot roots must differ")
            _ensure_shared_scheduler(shared_root, shared_capacity)

        campaign_lease: _SlotLease | None = None
        shared_lease: _SlotLease | None = None
        started = time.monotonic()
        campaign_first = _slot_first(
            opportunity_root,
            scope="campaign",
            capacity=self.max_parallel_evaluators,
        )
        shared_first = (
            _slot_first(
                opportunity_root,
                scope="shared",
                capacity=shared_capacity,
            )
            if shared_capacity is not None
            else None
        )
        try:
            while campaign_lease is None:
                campaign_lease = _try_acquire_slot(
                    root=campaign_root,
                    capacity=self.max_parallel_evaluators,
                    first=campaign_first,
                    opportunity_root=opportunity_root,
                    scope="campaign",
                )
                if campaign_lease is None:
                    time.sleep(0.5)
                    continue
                if shared_root is not None and shared_capacity is not None:
                    assert shared_first is not None
                    shared_lease = _try_acquire_slot(
                        root=shared_root,
                        capacity=shared_capacity,
                        first=shared_first,
                        opportunity_root=opportunity_root,
                        scope="shared_local_host",
                    )
                    if shared_lease is None:
                        # Never occupy a campaign lease while queued globally.
                        _release_slot(campaign_lease)
                        campaign_lease = None
                        time.sleep(0.5)
            (opportunity_root / "evaluator-queue.json").write_text(
                json.dumps(
                    {
                        "schema_version": "2.0",
                        "wait_seconds": time.monotonic() - started,
                        # Preserve the old field for existing dashboards.
                        "slot": str(campaign_lease.path),
                        "campaign_slot": str(campaign_lease.path),
                        "shared_slot": (
                            str(shared_lease.path)
                            if shared_lease is not None
                            else None
                        ),
                        "shared_capacity": shared_capacity,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            yield
        finally:
            _release_slot(shared_lease)
            _release_slot(campaign_lease)

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
        with self._evaluation_slot(opportunity_root):
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


def make_command_evaluator(
    *,
    task: TaskSpec,
    support_source: Path,
    repo_root: Path,
    python_bin: str,
    slot_root: Path | None = None,
    max_parallel_evaluators: int | None = None,
    shared_slot_root: Path | None = None,
    max_shared_parallel_evaluators: int | None = None,
) -> CommandEvaluator:
    """Construct the evaluator transport frozen by the task specification."""

    from .spec import ExecutionBackend

    arguments = {
        "task": task,
        "support_source": support_source,
        "repo_root": repo_root,
        "python_bin": python_bin,
        "slot_root": slot_root,
        "max_parallel_evaluators": max_parallel_evaluators,
        "shared_slot_root": shared_slot_root,
        "max_shared_parallel_evaluators": max_shared_parallel_evaluators,
    }
    if task.preferred_backend is ExecutionBackend.HYBRID_MODAL:
        from .hybrid_evaluator import ModalCommandEvaluator

        return ModalCommandEvaluator(**arguments)
    return CommandEvaluator(**arguments)

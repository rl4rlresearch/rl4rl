"""Frozen serial and parallel campaign-level execution orders."""

from __future__ import annotations

import fcntl
import json
import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .runner import run_one_opportunity
from .spec import (
    PARALLEL_EXECUTION_RULE,
    SERIAL_EXECUTION_RULE,
    ExecutionBackend,
    FactorialSpec,
    FrameworkSpec,
    TaskSpec,
)
from .state import SearchController, append_jsonl, utc_now


class CampaignLockedError(RuntimeError):
    """Another campaign-level writer already owns this campaign."""


class ParallelWaveError(RuntimeError):
    """One or more concurrently launched factorial opportunities failed."""


@dataclass(frozen=True)
class NextRun:
    run_id: str
    condition: str
    block: int
    order: int
    opportunity: int


@dataclass(frozen=True)
class ParallelWave:
    """The next least-advanced block wave under protocol 1.1."""

    block: int
    opportunity: int
    factorial_runs: tuple[NextRun, ...]
    no_search_run: NextRun | None
    recovery_subset: bool


def _schedule(campaign: Path) -> list[dict[str, object]]:
    value = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("campaign schedule must be a list")
    return sorted(value, key=lambda row: (int(row["block"]), int(row["order"])))


@contextmanager
def campaign_lock(campaign_dir: str | Path) -> Iterator[None]:
    """Hold the single-writer boundary for a whole campaign operation."""

    campaign = Path(campaign_dir).resolve()
    path = campaign / ".campaign-runner.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignLockedError(
                f"campaign already has an active writer: {campaign}"
            ) from error
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "acquired_at": datetime.now(UTC).isoformat(),
            },
            handle,
        )
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _next_run_from(
    assignment: dict[str, object], controller: SearchController
) -> NextRun:
    return NextRun(
        run_id=controller.state.run_id,
        condition=str(assignment["condition"]),
        block=int(assignment["block"]),
        order=int(assignment["order"]),
        opportunity=controller.state.next_opportunity,
    )


def next_run(campaign_dir: str | Path, spec: FactorialSpec) -> NextRun | None:
    """Choose the least-advanced run, then frozen block/order.

    This produces one opportunity per run per round. It balances provider drift,
    thermal/load effects, and operator timing across conditions more closely than
    completing all 100 opportunities of one condition before starting the next.
    """

    if spec.execution_rule != SERIAL_EXECUTION_RULE:
        raise ValueError(
            "serial campaign commands require "
            f"execution_rule={SERIAL_EXECUTION_RULE!r}"
        )
    campaign = Path(campaign_dir).resolve()
    eligible: list[tuple[int, int, int, dict[str, object], SearchController]] = []
    for assignment in _schedule(campaign):
        controller = SearchController.load(
            campaign / "runs" / str(assignment["run_id"]), spec
        )
        if controller.state.active is not None:
            raise RuntimeError(
                f"{controller.state.run_id} has an interrupted active opportunity; "
                "recover it explicitly before campaign execution"
            )
        if controller.state.status == "completed":
            continue
        eligible.append(
            (
                controller.state.proposals_used,
                int(assignment["block"]),
                int(assignment["order"]),
                assignment,
                controller,
            )
        )
    if not eligible:
        return None
    _, block, order, assignment, controller = min(
        eligible, key=lambda row: row[:3]
    )
    return _next_run_from(assignment, controller)


def next_parallel_wave(
    campaign_dir: str | Path, spec: FactorialSpec
) -> ParallelWave | None:
    """Select the next synchronized C0-C3 block wave.

    All non-completed runs at the campaign-wide minimum opportunity count are
    eligible. Within the earliest eligible block, C0-C3 cells launch together;
    N0 is selected at the same boundary but is always executed afterward. If an
    interrupted process launched only part of a group, the same rule selects
    only its still-lagging cells, making recovery deterministic and visible.
    """

    if spec.execution_rule != PARALLEL_EXECUTION_RULE:
        raise ValueError(
            "parallel campaign commands require "
            f"execution_rule={PARALLEL_EXECUTION_RULE!r}"
        )
    campaign = Path(campaign_dir).resolve()
    eligible: list[tuple[dict[str, object], SearchController]] = []
    controllers_by_block: dict[int, list[SearchController]] = {}
    for assignment in _schedule(campaign):
        controller = SearchController.load(
            campaign / "runs" / str(assignment["run_id"]), spec
        )
        controllers_by_block.setdefault(int(assignment["block"]), []).append(
            controller
        )
        if controller.state.active is not None:
            raise RuntimeError(
                f"{controller.state.run_id} has an interrupted active opportunity; "
                "recover it explicitly before campaign execution"
            )
        if controller.state.status != "completed":
            eligible.append((assignment, controller))
    if not eligible:
        return None

    minimum = min(controller.state.proposals_used for _, controller in eligible)
    block = min(
        int(assignment["block"])
        for assignment, controller in eligible
        if controller.state.proposals_used == minimum
    )
    selected = [
        (assignment, controller)
        for assignment, controller in eligible
        if int(assignment["block"]) == block
        and controller.state.proposals_used == minimum
    ]
    factorial = tuple(
        _next_run_from(assignment, controller)
        for assignment, controller in selected
        if str(assignment["condition"]) != "N0"
    )
    no_search = next(
        (
            _next_run_from(assignment, controller)
            for assignment, controller in selected
            if str(assignment["condition"]) == "N0"
        ),
        None,
    )
    recovery_subset = any(
        not controller.state.no_search
        and controller.state.proposals_used > minimum
        for controller in controllers_by_block[block]
    )
    return ParallelWave(
        block=block,
        opportunity=minimum + 1,
        factorial_runs=factorial,
        no_search_run=no_search,
        recovery_subset=recovery_subset,
    )


def _execute_parallel_wave(
    campaign: Path,
    wave: ParallelWave,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
    python_bin: str,
    codex_binary: str,
    codex_timeout_seconds: int,
) -> dict[str, object]:
    if task.preferred_backend is not ExecutionBackend.LOCAL:
        raise ValueError("parallel condition rounds require a local task backend")
    events_path = campaign / "parallel-rounds.jsonl"
    base_event = {
        "schema_version": "1.0",
        "execution_rule": spec.execution_rule,
        "block": wave.block,
        "opportunity": wave.opportunity,
        "factorial_run_ids": [run.run_id for run in wave.factorial_runs],
        "no_search_run_id": (
            wave.no_search_run.run_id if wave.no_search_run is not None else None
        ),
        "recovery_subset": wave.recovery_subset,
    }
    append_jsonl(
        events_path,
        {**base_event, "event": "parallel_wave_started", "timestamp": utc_now()},
    )

    barrier = (
        threading.Barrier(len(wave.factorial_runs))
        if len(wave.factorial_runs) > 1
        else None
    )

    def execute(
        selected: NextRun, *, synchronize_start: bool = False
    ) -> dict[str, object]:
        if synchronize_start and barrier is not None:
            barrier.wait()
        return run_one_opportunity(
            campaign / "runs" / selected.run_id,
            spec=spec,
            task=task,
            framework=framework,
            repo_root=repo_root,
            python_bin=python_bin,
            codex_binary=codex_binary,
            codex_timeout_seconds=codex_timeout_seconds,
        )

    factorial_records: list[dict[str, object]] = []
    failures: list[str] = []
    if wave.factorial_runs:
        with ThreadPoolExecutor(
            max_workers=len(wave.factorial_runs),
            thread_name_prefix="c0c3-parallel",
        ) as executor:
            futures = [
                executor.submit(execute, run, synchronize_start=True)
                for run in wave.factorial_runs
            ]
            for selected, future in zip(wave.factorial_runs, futures, strict=True):
                try:
                    factorial_records.append(future.result())
                except Exception as error:  # noqa: BLE001 - preserve all peer results
                    failures.append(
                        f"{selected.run_id}: {type(error).__name__}: {error}"
                    )
    if failures:
        append_jsonl(
            events_path,
            {
                **base_event,
                "event": "parallel_wave_failed",
                "timestamp": utc_now(),
                "errors": failures,
            },
        )
        raise ParallelWaveError("; ".join(failures))

    no_search_record = None
    if wave.no_search_run is not None:
        no_search_record = execute(wave.no_search_run)
    result = {
        **base_event,
        "factorial_records": factorial_records,
        "no_search_record": no_search_record,
    }
    append_jsonl(
        events_path,
        {**base_event, "event": "parallel_wave_completed", "timestamp": utc_now()},
    )
    return result


def run_parallel_next(
    campaign_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
    python_bin: str,
    codex_binary: str = "codex",
    codex_timeout_seconds: int = 3600,
) -> dict[str, object] | None:
    """Execute one version-1.1 block wave under the campaign writer lock."""

    campaign = Path(campaign_dir).resolve()
    with campaign_lock(campaign):
        wave = next_parallel_wave(campaign, spec)
        if wave is None:
            return None
        return _execute_parallel_wave(
            campaign,
            wave,
            spec=spec,
            task=task,
            framework=framework,
            repo_root=repo_root,
            python_bin=python_bin,
            codex_binary=codex_binary,
            codex_timeout_seconds=codex_timeout_seconds,
        )


def run_parallel_campaign(
    campaign_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
    python_bin: str,
    codex_binary: str = "codex",
    codex_timeout_seconds: int = 3600,
    max_block_rounds: int | None = None,
) -> Iterator[dict[str, object]]:
    """Yield completed version-1.1 waves while retaining one writer lock."""

    if max_block_rounds is not None and max_block_rounds < 1:
        raise ValueError("max_block_rounds must be positive")
    campaign = Path(campaign_dir).resolve()
    with campaign_lock(campaign):
        completed = 0
        while max_block_rounds is None or completed < max_block_rounds:
            wave = next_parallel_wave(campaign, spec)
            if wave is None:
                return
            yield _execute_parallel_wave(
                campaign,
                wave,
                spec=spec,
                task=task,
                framework=framework,
                repo_root=repo_root,
                python_bin=python_bin,
                codex_binary=codex_binary,
                codex_timeout_seconds=codex_timeout_seconds,
            )
            completed += 1

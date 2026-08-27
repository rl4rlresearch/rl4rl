"""Frozen serial and parallel campaign-level execution orders."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .agent_scheduler import WorkerQueueCancelled
from .prompts import (
    FROZEN_ASSUMPTION_PROMPT,
    FROZEN_ASSUMPTION_PROMPT_MANIFEST,
    artifact_clean_assumption_prompt_source,
)
from .runner import run_one_opportunity
from .spec import (
    INDIVIDUAL_EXECUTION_RULES,
    PARALLEL_EXECUTION_RULE,
    SERIAL_EXECUTION_RULE,
    STAGED_EXECUTION_RULES,
    STAGED_INDEPENDENT_EXECUTION_RULE,
    STAGED_PARALLEL_EXECUTION_RULE,
    ExecutionBackend,
    FactorialSpec,
    FrameworkSpec,
    TaskSpec,
)
from .state import SearchController, append_jsonl, atomic_json, utc_now
from .v3 import (
    mirror_shared_prefix,
    pair_for_run,
    pair_lock,
    validate_prompt_bundle,
)


class CampaignLockedError(RuntimeError):
    """Another campaign-level writer already owns this campaign."""


class ParallelWaveError(RuntimeError):
    """One or more concurrently launched factorial opportunities failed."""


class IndependentTrajectoryError(RuntimeError):
    """An independently advancing trajectory stopped the staged launcher."""


class TrajectoryLockedError(RuntimeError):
    """Another individually controlled process already owns this trajectory."""


class StageGateLockedError(RuntimeError):
    """Another process is momentarily checking or changing a stage gate."""


@dataclass(frozen=True)
class NextRun:
    run_id: str
    condition: str
    block: int
    order: int
    opportunity: int


@dataclass(frozen=True)
class ParallelWave:
    """One synchronized factorial or serial N0 wave."""

    block: int
    opportunity: int
    factorial_runs: tuple[NextRun, ...]
    no_search_run: NextRun | None
    recovery_subset: bool
    execution_stage: str


FACTORIAL_STAGE = "factorial"
NO_SEARCH_STAGE = "no-search"
STAGED_EXECUTION_STAGES = frozenset({FACTORIAL_STAGE, NO_SEARCH_STAGE})


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


@contextmanager
def trajectory_lock(run_dir: str | Path) -> Iterator[None]:
    """Own one individually controlled trajectory, without blocking its peers."""

    run = Path(run_dir).resolve()
    path = run / ".trajectory-controller.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TrajectoryLockedError(
                f"trajectory already has an active controller: {run}"
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


@contextmanager
def stage_gate_lock(campaign_dir: str | Path) -> Iterator[None]:
    """Serialize brief eligibility checks while allowing peers to queue."""

    campaign = Path(campaign_dir).resolve()
    path = campaign / ".stage-gate.lock"
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
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
            f"serial campaign commands require execution_rule={SERIAL_EXECUTION_RULE!r}"
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
    _, block, order, assignment, controller = min(eligible, key=lambda row: row[:3])
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
        controllers_by_block.setdefault(int(assignment["block"]), []).append(controller)
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
        not controller.state.no_search and controller.state.proposals_used > minimum
        for controller in controllers_by_block[block]
    )
    return ParallelWave(
        block=block,
        opportunity=minimum + 1,
        factorial_runs=factorial,
        no_search_run=no_search,
        recovery_subset=recovery_subset,
        execution_stage="synchronized-all-runs",
    )


def _staged_stage_assignments(
    campaign_dir: str | Path,
    spec: FactorialSpec,
    *,
    block: int,
    stage: str,
    reject_active: bool = True,
) -> list[tuple[dict[str, object], SearchController]]:
    """Load a valid, explicit stage without deciding its execution geometry."""

    if spec.execution_rule not in STAGED_EXECUTION_RULES:
        raise ValueError("staged campaign commands require a staged execution rule")
    if block < 1 or block > spec.blocks:
        raise ValueError(f"block must be between 1 and {spec.blocks}")
    if stage not in STAGED_EXECUTION_STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGED_EXECUTION_STAGES)}")
    if stage == NO_SEARCH_STAGE and not spec.include_no_search:
        raise ValueError("this protocol has no N0 no-search stage")

    campaign = Path(campaign_dir).resolve()
    selected: list[tuple[dict[str, object], SearchController]] = []
    all_factorial: list[SearchController] = []
    primary_factorial: list[SearchController] = []
    for assignment in _schedule(campaign):
        controller = SearchController.load(
            campaign / "runs" / str(assignment["run_id"]), spec
        )
        if reject_active and controller.state.active is not None:
            raise RuntimeError(
                f"{controller.state.run_id} has an interrupted active opportunity; "
                "recover it explicitly before campaign execution"
            )
        assignment_block = int(assignment["block"])
        is_no_search = str(assignment["condition"]) == "N0"
        if assignment_block == 1 and not is_no_search:
            primary_factorial.append(controller)
        if assignment_block != block:
            continue
        if not is_no_search:
            all_factorial.append(controller)
        if (stage == NO_SEARCH_STAGE) != is_no_search:
            continue
        selected.append((assignment, controller))

    if stage == NO_SEARCH_STAGE:
        required_factorial = primary_factorial + [
            controller
            for controller in all_factorial
            if controller not in primary_factorial
        ]
        if any(
            controller.state.status != "completed" for controller in required_factorial
        ):
            raise ValueError(
                "complete the frozen primary stage and this block's factorial "
                "stage before starting N0"
            )
    if (campaign / "sealed-layer-b").exists() or (campaign / "sealed-layer-c").exists():
        raise ValueError(
            "optional extensions must be activated before primary Layer B/C "
            "outputs are unsealed; new trajectories cannot begin after those "
            "outputs are unsealed"
        )

    expected_count = 4 if stage == FACTORIAL_STAGE else 1
    if len(selected) != expected_count:
        raise ValueError(
            f"block {block} stage {stage!r} requires {expected_count} scheduled "
            f"runs, found {len(selected)}"
        )
    return selected


def next_staged_parallel_wave(
    campaign_dir: str | Path,
    spec: FactorialSpec,
    *,
    block: int,
    stage: str,
) -> ParallelWave | None:
    """Select one protocol-1.3 synchronized wave from an explicit stage."""

    if spec.execution_rule != STAGED_PARALLEL_EXECUTION_RULE:
        raise ValueError(
            "staged parallel campaign commands require "
            f"execution_rule={STAGED_PARALLEL_EXECUTION_RULE!r}"
        )
    selected = _staged_stage_assignments(
        campaign_dir,
        spec,
        block=block,
        stage=stage,
    )
    eligible = [
        (assignment, controller)
        for assignment, controller in selected
        if controller.state.status != "completed"
    ]
    if not eligible:
        return None

    minimum = min(controller.state.proposals_used for _, controller in eligible)
    at_minimum = [
        (assignment, controller)
        for assignment, controller in eligible
        if controller.state.proposals_used == minimum
    ]
    factorial = tuple(
        _next_run_from(assignment, controller)
        for assignment, controller in at_minimum
        if not controller.state.no_search
    )
    no_search = next(
        (
            _next_run_from(assignment, controller)
            for assignment, controller in at_minimum
            if controller.state.no_search
        ),
        None,
    )
    recovery_subset = stage == FACTORIAL_STAGE and any(
        controller.state.proposals_used > minimum for _, controller in selected
    )
    return ParallelWave(
        block=block,
        opportunity=minimum + 1,
        factorial_runs=factorial,
        no_search_run=no_search,
        recovery_subset=recovery_subset,
        execution_stage=f"block-{block:02d}-{stage}",
    )


def staged_independent_trajectories(
    campaign_dir: str | Path,
    spec: FactorialSpec,
    *,
    block: int,
    stage: str,
) -> tuple[NextRun, ...]:
    """Select every unfinished trajectory in a protocol-1.4 stage.

    Unlike a synchronized wave, each selected trajectory continues through its
    entire remaining budget in its own worker. The initial barrier is only a
    simultaneous launch boundary; it is not repeated between opportunities.
    """

    if spec.execution_rule != STAGED_INDEPENDENT_EXECUTION_RULE:
        raise ValueError(
            "staged independent campaign commands require "
            f"execution_rule={STAGED_INDEPENDENT_EXECUTION_RULE!r}"
        )
    selected = _staged_stage_assignments(
        campaign_dir,
        spec,
        block=block,
        stage=stage,
    )
    return tuple(
        _next_run_from(assignment, controller)
        for assignment, controller in selected
        if controller.state.status != "completed"
    )


def _individual_assignment(
    campaign_dir: str | Path,
    spec: FactorialSpec,
    *,
    run_id: str,
) -> tuple[Path, dict[str, object], str]:
    """Resolve one individually controlled v1.5 run from immutable schedule data."""

    if spec.execution_rule not in INDIVIDUAL_EXECUTION_RULES:
        raise ValueError(
            "individually controlled trajectory commands require "
            f"one of execution_rule={sorted(INDIVIDUAL_EXECUTION_RULES)!r}"
        )
    campaign = Path(campaign_dir).resolve()
    assignment = next(
        (row for row in _schedule(campaign) if str(row["run_id"]) == run_id),
        None,
    )
    if assignment is None:
        raise ValueError(f"run ID is not in the campaign schedule: {run_id}")
    stage = NO_SEARCH_STAGE if str(assignment["condition"]) == "N0" else FACTORIAL_STAGE
    return campaign, assignment, stage


def _append_trajectory_lifecycle(
    campaign: Path,
    run_dir: Path,
    *,
    event: str,
    assignment: dict[str, object],
    stage: str,
    **details: object,
) -> None:
    """Write append-only lifecycle provenance without mutating scientific state."""

    record = {
        "schema_version": "1.0",
        "event": event,
        "timestamp": utc_now(),
        "run_id": str(assignment["run_id"]),
        "condition": str(assignment["condition"]),
        "block": int(assignment["block"]),
        "stage": stage,
        **details,
    }
    append_jsonl(campaign / "trajectory-lifecycle.jsonl", record)
    append_jsonl(run_dir / "lifecycle.jsonl", record)


def _pause_request_path(run_dir: Path) -> Path:
    return run_dir / "pause-request.json"


def _freeze_artifact_clean_assumption_prompt(
    *,
    campaign: Path,
    run_dir: Path,
    repo_root: Path,
    framework: FrameworkSpec,
) -> dict[str, object] | None:
    """Snapshot the live operator prompt exactly once, at trajectory start."""

    source = artifact_clean_assumption_prompt_source(
        campaign=campaign,
        repo_root=repo_root,
        framework=framework,
    )
    if source is None:
        return None
    target = run_dir / FROZEN_ASSUMPTION_PROMPT
    manifest_path = run_dir / FROZEN_ASSUMPTION_PROMPT_MANIFEST
    if target.exists() or manifest_path.exists():
        raise RuntimeError(
            "an unstarted trajectory already has a subject-prompt snapshot"
        )
    content = source.read_bytes()
    prompt_sha256 = hashlib.sha256(content).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=False)
    temporary = target.with_name(f".{target.name}.partial-{uuid.uuid4().hex}")
    temporary.write_bytes(content)
    os.replace(temporary, target)
    manifest = {
        "schema_version": "1.0",
        "frozen_at": utc_now(),
        "prompt_profile": framework.prompt_profile,
        "source_path": str(source),
        "sha256": prompt_sha256,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _take_pause_request(run_dir: Path) -> dict[str, object] | None:
    """Atomically claim a request so a concurrent request is never silently lost."""

    request = _pause_request_path(run_dir)
    claimed = run_dir / f".pause-request-consumed-{uuid.uuid4().hex}.json"
    try:
        os.replace(request, claimed)
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(claimed.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {"reason": "unreadable pause request"}
    finally:
        claimed.unlink(missing_ok=True)
    return payload if isinstance(payload, dict) else {"reason": "invalid pause request"}


def request_staged_trajectory_pause(
    campaign_dir: str | Path,
    *,
    spec: FactorialSpec,
    run_id: str,
    reason: str,
) -> dict[str, object]:
    """Request a cooperative v1.5 pause after the current opportunity commits."""

    if not reason.strip():
        raise ValueError("pause reason cannot be blank")
    campaign, assignment, stage = _individual_assignment(
        campaign_dir, spec, run_id=run_id
    )
    run_dir = campaign / "runs" / run_id
    controller = SearchController.load(run_dir, spec)
    if controller.state.status == "completed":
        raise ValueError("completed trajectories cannot be paused")
    request = {
        "schema_version": "1.0",
        "run_id": run_id,
        "requested_at": utc_now(),
        "reason": reason.strip(),
    }
    atomic_json(_pause_request_path(run_dir), request)
    _append_trajectory_lifecycle(
        campaign,
        run_dir,
        event="trajectory_pause_requested",
        assignment=assignment,
        stage=stage,
        reason=reason.strip(),
        active_opportunity=(
            controller.state.active.index
            if controller.state.active is not None
            else None
        ),
    )
    return {
        "run_id": run_id,
        "status": "pause_requested",
        "active_opportunity": (
            controller.state.active.index
            if controller.state.active is not None
            else None
        ),
    }


def run_v3_paired_opportunity(
    campaign_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
    python_bin: str,
    run_id: str,
    codex_binary: str = "codex",
    codex_timeout_seconds: int = 3600,
) -> dict[str, object]:
    """Advance a v3 run, physically sharing each pre-intervention pair prefix."""

    if spec.protocol_version != "3.0":
        raise ValueError("paired-prefix execution requires protocol 3.0")
    campaign = Path(campaign_dir).resolve()
    validate_prompt_bundle(campaign, spec=spec, framework=framework)
    pair = pair_for_run(campaign, run_id)
    with pair_lock(campaign, pair):
        leader_dir = campaign / "runs" / str(pair["leader_run_id"])
        shadow_dir = campaign / "runs" / str(pair["shadow_run_id"])
        leader = SearchController.load(leader_dir, spec)
        shadow = SearchController.load(shadow_dir, spec)
        fork = int(pair["fork_opportunity"])
        if (
            leader.state.proposals_used < fork - 1
            or shadow.state.proposals_used < fork - 1
        ):
            if leader.state.proposals_used == shadow.state.proposals_used + 1:
                mirrored = mirror_shared_prefix(campaign, spec=spec, pair=pair)
                return {
                    "schema_version": "3.0",
                    "requested_run_id": run_id,
                    "shared_prefix": True,
                    "recovered_pending_mirror": True,
                    "mirror": mirrored,
                }
            if leader.state.proposals_used != shadow.state.proposals_used:
                raise RuntimeError("v3 paired-prefix states diverged before fork")
            record = run_one_opportunity(
                leader_dir,
                spec=spec,
                task=task,
                framework=framework,
                repo_root=repo_root,
                python_bin=python_bin,
                codex_binary=codex_binary,
                codex_timeout_seconds=codex_timeout_seconds,
                allow_v3_prefix_leader=True,
            )
            mirrored = mirror_shared_prefix(campaign, spec=spec, pair=pair)
            return {
                **record,
                "requested_run_id": run_id,
                "physical_run_id": pair["leader_run_id"],
                "shared_prefix": True,
                "resource_accounting": "shared_prefix_charge_once_to_pair",
                "mirror": mirrored,
            }
    return run_one_opportunity(
        campaign / "runs" / run_id,
        spec=spec,
        task=task,
        framework=framework,
        repo_root=repo_root,
        python_bin=python_bin,
        codex_binary=codex_binary,
        codex_timeout_seconds=codex_timeout_seconds,
    )


def run_staged_individual_trajectory(
    campaign_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
    python_bin: str,
    run_id: str,
    resume: bool = False,
    codex_binary: str = "codex",
    codex_timeout_seconds: int = 3600,
) -> dict[str, object]:
    """Start or resume one v1.5 trajectory without owning any peer trajectory.

    A pause is cooperative: a request is observed only before the next proposal
    begins, so an in-flight Codex/evaluator call finishes and is recorded normally.
    """

    if task.preferred_backend not in {
        ExecutionBackend.LOCAL,
        ExecutionBackend.HYBRID_MODAL,
    }:
        raise ValueError(
            "individually controlled trajectories require local Codex execution"
        )
    campaign, assignment, stage = _individual_assignment(
        campaign_dir, spec, run_id=run_id
    )
    run_dir = campaign / "runs" / run_id
    with trajectory_lock(run_dir):
        with stage_gate_lock(campaign):
            selected = _staged_stage_assignments(
                campaign,
                spec,
                block=int(assignment["block"]),
                stage=stage,
                reject_active=False,
            )
            selected_ids = {controller.state.run_id for _, controller in selected}
            if run_id not in selected_ids:
                raise RuntimeError("run is not eligible for its frozen stage")
            controller = SearchController.load(run_dir, spec)
            if controller.state.active is not None:
                raise RuntimeError(
                    "run has an interrupted active opportunity; recover it explicitly "
                    "before starting or resuming"
                )
            if controller.state.status == "completed":
                return {
                    "run_id": run_id,
                    "condition": str(assignment["condition"]),
                    "status": "completed",
                    "proposals_used": controller.state.proposals_used,
                    "completed_opportunities": 0,
                    "stop_reason": "already_completed",
                }
            if not resume and controller.state.proposals_used != 0:
                raise ValueError(
                    "an already-started trajectory must use resume-staged-trajectory"
                )
            if resume:
                cleared = _take_pause_request(run_dir)
                if cleared is not None:
                    _append_trajectory_lifecycle(
                        campaign,
                        run_dir,
                        event="trajectory_pause_cleared_for_resume",
                        assignment=assignment,
                        stage=stage,
                        prior_reason=str(cleared.get("reason", "[not recorded]")),
                    )
            elif _pause_request_path(run_dir).exists():
                raise RuntimeError(
                    "a pause request is pending; use resume-staged-trajectory to "
                    "clear it"
                )
            if spec.protocol_version == "3.0":
                prompt_snapshot = validate_prompt_bundle(
                    campaign, spec=spec, framework=framework
                )
            else:
                prompt_snapshot = (
                    _freeze_artifact_clean_assumption_prompt(
                        campaign=campaign,
                        run_dir=run_dir,
                        repo_root=repo_root,
                        framework=framework,
                    )
                    if not resume and controller.state.proposals_used == 0
                    else None
                )
            _append_trajectory_lifecycle(
                campaign,
                run_dir,
                event=("trajectory_resumed" if resume else "trajectory_started"),
                assignment=assignment,
                stage=stage,
                starting_opportunity=controller.state.next_opportunity,
                assumption_prompt_sha256=(
                    prompt_snapshot.get("sha256")
                    or prompt_snapshot.get("bundle_sha256")
                    if prompt_snapshot is not None
                    else None
                ),
            )

        completed_opportunities = 0
        while True:
            controller = SearchController.load(run_dir, spec)
            if controller.state.active is not None:
                raise RuntimeError(
                    "run has an interrupted active opportunity; recover it explicitly"
                )
            if controller.state.status == "completed":
                outcome = {
                    "run_id": run_id,
                    "condition": str(assignment["condition"]),
                    "status": "completed",
                    "proposals_used": controller.state.proposals_used,
                    "completed_opportunities": completed_opportunities,
                    "stop_reason": "budget_completed",
                }
                _append_trajectory_lifecycle(
                    campaign,
                    run_dir,
                    event="trajectory_completed",
                    assignment=assignment,
                    stage=stage,
                    **outcome,
                )
                return outcome
            pause_request = _take_pause_request(run_dir)
            if pause_request is not None:
                outcome = {
                    "run_id": run_id,
                    "condition": str(assignment["condition"]),
                    "status": "paused",
                    "proposals_used": controller.state.proposals_used,
                    "completed_opportunities": completed_opportunities,
                    "stop_reason": "cooperative_pause",
                }
                _append_trajectory_lifecycle(
                    campaign,
                    run_dir,
                    event="trajectory_paused",
                    assignment=assignment,
                    stage=stage,
                    reason=str(pause_request.get("reason", "[not recorded]")),
                    **outcome,
                )
                return outcome
            try:
                record = (
                    run_v3_paired_opportunity(
                        campaign,
                        spec=spec,
                        task=task,
                        framework=framework,
                        repo_root=repo_root,
                        python_bin=python_bin,
                        run_id=run_id,
                        codex_binary=codex_binary,
                        codex_timeout_seconds=codex_timeout_seconds,
                    )
                    if spec.protocol_version == "3.0"
                    else run_one_opportunity(
                        run_dir,
                        spec=spec,
                        task=task,
                        framework=framework,
                        repo_root=repo_root,
                        python_bin=python_bin,
                        codex_binary=codex_binary,
                        codex_timeout_seconds=codex_timeout_seconds,
                    )
                )
            except WorkerQueueCancelled:
                # No proposal was begun and no scientific budget was consumed.
                # The next loop iteration claims the cooperative pause marker.
                continue
            except KeyboardInterrupt:
                _append_trajectory_lifecycle(
                    campaign,
                    run_dir,
                    event="trajectory_command_interrupted",
                    assignment=assignment,
                    stage=stage,
                    completed_opportunities=completed_opportunities,
                )
                raise
            completed_opportunities += 1
            controller = SearchController.load(run_dir, spec)
            if controller.state.status == "token_threshold_reached":
                outcome = {
                    "run_id": run_id,
                    "condition": str(assignment["condition"]),
                    "status": "token_threshold_reached",
                    "proposals_used": controller.state.proposals_used,
                    "completed_opportunities": completed_opportunities,
                    "stop_reason": "subject_visible_token_threshold",
                }
                _append_trajectory_lifecycle(
                    campaign,
                    run_dir,
                    event="trajectory_token_threshold_reached",
                    assignment=assignment,
                    stage=stage,
                    **outcome,
                )
                return outcome
            if (
                record.get("evaluation", {}).get("failure_kind") == "provider"
                and record.get("usage_increment", {}).get("total_tokens", 0) == 0
            ):
                outcome = {
                    "run_id": run_id,
                    "condition": str(assignment["condition"]),
                    "status": controller.state.status,
                    "proposals_used": controller.state.proposals_used,
                    "completed_opportunities": completed_opportunities,
                    "stop_reason": "provider_transport_failure",
                }
                _append_trajectory_lifecycle(
                    campaign,
                    run_dir,
                    event="trajectory_stopped_provider_failure",
                    assignment=assignment,
                    stage=stage,
                    **outcome,
                )
                return outcome


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
        "execution_stage": wave.execution_stage,
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

    provider_failures = [
        str(record.get("run_id", "unknown"))
        for record in factorial_records
        if record.get("evaluation", {}).get("failure_kind") == "provider"
        and record.get("usage_increment", {}).get("total_tokens", 0) == 0
    ]
    if provider_failures:
        message = (
            "Codex provider transport failed before token accounting for: "
            + ", ".join(provider_failures)
        )
        append_jsonl(
            events_path,
            {
                **base_event,
                "event": "parallel_wave_failed",
                "timestamp": utc_now(),
                "errors": [message],
            },
        )
        raise ParallelWaveError(message)

    no_search_record = None
    if wave.no_search_run is not None:
        no_search_record = execute(wave.no_search_run)
        if (
            no_search_record.get("evaluation", {}).get("failure_kind") == "provider"
            and no_search_record.get("usage_increment", {}).get("total_tokens", 0) == 0
        ):
            message = (
                "Codex provider transport failed before token accounting for: "
                f"{wave.no_search_run.run_id}"
            )
            append_jsonl(
                events_path,
                {
                    **base_event,
                    "event": "parallel_wave_failed",
                    "timestamp": utc_now(),
                    "errors": [message],
                },
            )
            raise ParallelWaveError(message)
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


def run_staged_next(
    campaign_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
    python_bin: str,
    block: int,
    stage: str,
    codex_binary: str = "codex",
    codex_timeout_seconds: int = 3600,
) -> dict[str, object] | None:
    """Execute one protocol-1.3 wave from an explicit frozen stage."""

    campaign = Path(campaign_dir).resolve()
    with campaign_lock(campaign):
        wave = next_staged_parallel_wave(
            campaign,
            spec,
            block=block,
            stage=stage,
        )
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


def run_staged_campaign(
    campaign_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
    python_bin: str,
    block: int,
    stage: str,
    codex_binary: str = "codex",
    codex_timeout_seconds: int = 3600,
    max_block_rounds: int | None = None,
) -> Iterator[dict[str, object]]:
    """Complete one protocol-1.3 stage without advancing dormant stages."""

    if max_block_rounds is not None and max_block_rounds < 1:
        raise ValueError("max_block_rounds must be positive")
    campaign = Path(campaign_dir).resolve()
    with campaign_lock(campaign):
        completed = 0
        while max_block_rounds is None or completed < max_block_rounds:
            wave = next_staged_parallel_wave(
                campaign,
                spec,
                block=block,
                stage=stage,
            )
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


def _run_independent_trajectory(
    campaign: Path,
    selected: NextRun,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
    python_bin: str,
    codex_binary: str,
    codex_timeout_seconds: int,
    stage: str,
    launch_barrier: threading.Barrier,
    abort: threading.Event,
    errors: list[str],
    errors_lock: threading.Lock,
) -> dict[str, object]:
    """Advance one run until its frozen budget completes or a hard failure occurs."""

    launch_barrier.wait()
    records: list[dict[str, object]] = []
    run_dir = campaign / "runs" / selected.run_id
    while not abort.is_set():
        controller = SearchController.load(run_dir, spec)
        if controller.state.active is not None:
            message = (
                f"{selected.run_id} has an interrupted active opportunity; "
                "recover it explicitly before campaign execution"
            )
            with errors_lock:
                errors.append(message)
            abort.set()
            break
        if controller.state.status == "completed":
            break
        try:
            record = run_one_opportunity(
                run_dir,
                spec=spec,
                task=task,
                framework=framework,
                repo_root=repo_root,
                python_bin=python_bin,
                codex_binary=codex_binary,
                codex_timeout_seconds=codex_timeout_seconds,
            )
        except Exception as error:  # noqa: BLE001 - preserve peer run state
            message = f"{selected.run_id}: {type(error).__name__}: {error}"
            with errors_lock:
                errors.append(message)
            abort.set()
            break
        records.append(record)
        if (
            record.get("evaluation", {}).get("failure_kind") == "provider"
            and record.get("usage_increment", {}).get("total_tokens", 0) == 0
        ):
            message = (
                "Codex provider transport failed before token accounting for: "
                f"{selected.run_id}"
            )
            with errors_lock:
                errors.append(message)
            abort.set()
            break

    final_state = SearchController.load(run_dir, spec).state
    return {
        "run_id": selected.run_id,
        "condition": selected.condition,
        "block": selected.block,
        "execution_stage": f"block-{selected.block:02d}-{stage}-independent",
        "starting_opportunity": selected.opportunity,
        "completed_opportunities": len(records),
        "status": final_state.status,
        "proposals_used": final_state.proposals_used,
    }


def run_staged_independent_campaign(
    campaign_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
    python_bin: str,
    block: int,
    stage: str,
    codex_binary: str = "codex",
    codex_timeout_seconds: int = 3600,
) -> Iterator[dict[str, object]]:
    """Run every selected protocol-1.4 trajectory independently and concurrently.

    The campaign writer starts all selected trajectories behind one initial
    barrier. Each worker then immediately starts its next opportunity when its
    own evaluator finishes; no worker waits for a peer's opportunity boundary.
    A provider transport failure or unexpected runner exception stops new work
    after already in-flight opportunities finish, preserving their charged
    records for explicit recovery or resumption.
    """

    if spec.execution_rule != STAGED_INDEPENDENT_EXECUTION_RULE:
        raise ValueError(
            "staged independent campaign commands require "
            f"execution_rule={STAGED_INDEPENDENT_EXECUTION_RULE!r}"
        )
    if task.preferred_backend is not ExecutionBackend.LOCAL:
        raise ValueError(
            "independent parallel trajectories currently require a local task backend"
        )
    campaign = Path(campaign_dir).resolve()
    with campaign_lock(campaign):
        selected_runs = staged_independent_trajectories(
            campaign,
            spec,
            block=block,
            stage=stage,
        )
        if not selected_runs:
            return
        events_path = campaign / "independent-trajectories.jsonl"
        base_event = {
            "schema_version": "1.0",
            "execution_rule": spec.execution_rule,
            "execution_stage": f"block-{block:02d}-{stage}",
            "block": block,
            "stage": stage,
            "run_ids": [run.run_id for run in selected_runs],
            "starting_opportunities": {
                run.run_id: run.opportunity for run in selected_runs
            },
        }
        append_jsonl(
            events_path,
            {
                **base_event,
                "event": "independent_trajectory_batch_started",
                "timestamp": utc_now(),
            },
        )
        abort = threading.Event()
        errors: list[str] = []
        errors_lock = threading.Lock()
        launch_barrier = threading.Barrier(len(selected_runs))
        outcomes: list[dict[str, object]] = []
        with ThreadPoolExecutor(
            max_workers=len(selected_runs),
            thread_name_prefix="c0c3-independent",
        ) as executor:
            futures = [
                executor.submit(
                    _run_independent_trajectory,
                    campaign,
                    selected,
                    spec=spec,
                    task=task,
                    framework=framework,
                    repo_root=repo_root,
                    python_bin=python_bin,
                    codex_binary=codex_binary,
                    codex_timeout_seconds=codex_timeout_seconds,
                    stage=stage,
                    launch_barrier=launch_barrier,
                    abort=abort,
                    errors=errors,
                    errors_lock=errors_lock,
                )
                for selected in selected_runs
            ]
            for future in as_completed(futures):
                outcome = future.result()
                outcomes.append(outcome)
                append_jsonl(
                    events_path,
                    {
                        **base_event,
                        "event": (
                            "independent_trajectory_completed"
                            if outcome["status"] == "completed"
                            else "independent_trajectory_stopped"
                        ),
                        "timestamp": utc_now(),
                        "trajectory": outcome,
                    },
                )
                yield outcome
        if errors:
            append_jsonl(
                events_path,
                {
                    **base_event,
                    "event": "independent_trajectory_batch_failed",
                    "timestamp": utc_now(),
                    "errors": errors,
                    "trajectories": outcomes,
                },
            )
            raise IndependentTrajectoryError("; ".join(errors))
        append_jsonl(
            events_path,
            {
                **base_event,
                "event": "independent_trajectory_batch_completed",
                "timestamp": utc_now(),
                "trajectories": outcomes,
            },
        )

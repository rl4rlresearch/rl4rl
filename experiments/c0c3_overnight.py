#!/usr/bin/env python3
"""Durable local supervisor for the currently active C0-C3 campaigns.

This is deliberately outside ``experiments/c0c3_factorial``.  Existing
campaigns hash that package as part of their scientific runtime, so operational
supervision must not alter the frozen controller used to create them.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = REPO_ROOT / "architecture_discovery/.venv/bin/python"
PROFILE = os.environ.get("RL4RL_OVERNIGHT_PROFILE", "primary")
if PROFILE == "primary":
    DEFAULT_CONTROL_ROOT = REPO_ROOT / "data/c0c3/overnight-control"
    DEFAULT_SCREEN_SESSION = "rl4rl-c0c3-overnight"
elif PROFILE == "1644-extension":
    DEFAULT_CONTROL_ROOT = REPO_ROOT / "data/c0c3/overnight-control-1644-extension"
    DEFAULT_SCREEN_SESSION = "rl4rl-c0c3-1644-extension"
elif PROFILE == "1644-confined":
    DEFAULT_CONTROL_ROOT = REPO_ROOT / "data/c0c3/overnight-control-1644-confined"
    DEFAULT_SCREEN_SESSION = "rl4rl-c0c3-1644-confined"
elif PROFILE == "1644-confined-fresh":
    DEFAULT_CONTROL_ROOT = (
        REPO_ROOT / "data/c0c3/overnight-control-1644-confined-fresh"
    )
    DEFAULT_SCREEN_SESSION = "rl4rl-c0c3-1644-confined-fresh"
elif PROFILE == "openevolve-v2":
    DEFAULT_CONTROL_ROOT = REPO_ROOT / "data/c0c3/overnight-control-openevolve-v2"
    DEFAULT_SCREEN_SESSION = "rl4rl-c0c3-openevolve-v2"
elif PROFILE == "autoresearch-v1.7":
    DEFAULT_CONTROL_ROOT = REPO_ROOT / "data/c0c3/overnight-control-autoresearch-v1-7"
    DEFAULT_SCREEN_SESSION = "rl4rl-c0c3-autoresearch-v1-7"
elif PROFILE == "openevolve-v2.1":
    DEFAULT_CONTROL_ROOT = REPO_ROOT / "data/c0c3/overnight-control-openevolve-v2-1"
    DEFAULT_SCREEN_SESSION = "rl4rl-c0c3-openevolve-v2-1"
else:
    raise RuntimeError(f"unknown overnight profile: {PROFILE}")
CONTROL_ROOT = Path(
    os.environ.get("RL4RL_OVERNIGHT_CONTROL_ROOT", str(DEFAULT_CONTROL_ROOT))
).expanduser().resolve()
DESIRED_PATH = CONTROL_ROOT / "desired.json"
STATUS_PATH = CONTROL_ROOT / "status.json"
SUPERVISOR_LOCK = CONTROL_ROOT / "supervisor.lock"
CONTROL_LOCK = CONTROL_ROOT / "control.lock"
SUPERVISOR_LOG = CONTROL_ROOT / "supervisor.log"
SCREEN_SESSION = os.environ.get(
    "RL4RL_OVERNIGHT_SCREEN_SESSION", DEFAULT_SCREEN_SESSION
)
SCHEMA_VERSION = "1.0"
POLL_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30 * 60


@dataclasses.dataclass(frozen=True)
class CampaignPlan:
    key: str
    runtime_root: Path
    campaign: Path
    mode: str
    blocks: tuple[int, ...] = ()
    pause_after_proposals: int | None = None


@dataclasses.dataclass(frozen=True)
class Job:
    key: str
    group: str
    runtime_root: Path
    campaign: Path
    mode: str
    run_id: str | None = None
    blocks: tuple[int, ...] = ()
    pause_after_proposals: int | None = None


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def plans(profile: str | None = None) -> tuple[CampaignPlan, ...]:
    """Return the intended overnight roster, excluding superseded campaigns."""

    selected_profile = profile or PROFILE
    if selected_profile == "1644-extension":
        return (
            CampaignPlan(
                key="autoresearch-v1.5-1644-extension",
                runtime_root=_env_path(
                    "RL4RL_V15_1644_RUNTIME",
                    Path("/private/tmp/rl4rl-c0c3-v15-codex1644"),
                ),
                campaign=_env_path(
                    "RL4RL_V15_1644_EXTENSION_CAMPAIGN",
                    Path(
                        "/private/tmp/"
                        "rl4rl-v15-codex1644-extension-campaign-live-20260822a"
                    ),
                ),
                mode="individual-trajectories",
                blocks=(2, 3),
            ),
        )
    if selected_profile == "1644-confined":
        return (
            CampaignPlan(
                key="autoresearch-v1.6-1644-confined",
                runtime_root=_env_path(
                    "RL4RL_V16_1644_RUNTIME",
                    Path("/private/tmp/rl4rl-c0c3-v16-confined"),
                ),
                campaign=_env_path(
                    "RL4RL_V16_1644_CAMPAIGN",
                    Path(
                        "/private/tmp/"
                        "rl4rl-v16-codex1644-confined-campaign-live-20260822b"
                    ),
                ),
                mode="individual-trajectories",
                blocks=(1, 2, 3),
            ),
        )
    if selected_profile == "1644-confined-fresh":
        return (
            CampaignPlan(
                key="autoresearch-v1.6-1644-confined-fresh",
                runtime_root=_env_path(
                    "RL4RL_V16_1644_FRESH_RUNTIME",
                    Path("/private/tmp/rl4rl-c0c3-v16-confined"),
                ),
                campaign=_env_path(
                    "RL4RL_V16_1644_FRESH_CAMPAIGN",
                    Path(
                        "/private/tmp/"
                        "rl4rl-v16-codex1644-confined-campaign-fresh-20260822c"
                    ),
                ),
                mode="individual-trajectories",
                blocks=(1, 2, 3),
                pause_after_proposals=100,
            ),
        )
    if selected_profile == "openevolve-v2":
        return (
            CampaignPlan(
                key="openevolve-v2",
                runtime_root=_env_path(
                    "RL4RL_OPENEVOLVE_V2_RUNTIME",
                    Path("/private/tmp/rl4rl-c0c3-openevolve-v2"),
                ),
                campaign=_env_path(
                    "RL4RL_OPENEVOLVE_V2_CAMPAIGN",
                    REPO_ROOT
                    / "data/c0c3/controlled-openevolve-transformer-v2-mps-campaign",
                ),
                mode="individual-trajectories",
                blocks=(1, 2, 3),
            ),
        )
    if selected_profile == "autoresearch-v1.7":
        return (
            CampaignPlan(
                key="autoresearch-v1.7",
                runtime_root=_env_path(
                    "RL4RL_AUTORESEARCH_V17_RUNTIME",
                    Path("/private/tmp/rl4rl-c0c3-autoresearch-v1-7"),
                ),
                campaign=_env_path(
                    "RL4RL_AUTORESEARCH_V17_CAMPAIGN",
                    REPO_ROOT
                    / "data/c0c3/transformer-optimization-v1-7-source-only-campaign",
                ),
                mode="individual-trajectories",
                blocks=(1, 2, 3),
            ),
        )
    if selected_profile == "openevolve-v2.1":
        return (
            CampaignPlan(
                key="openevolve-v2.1",
                runtime_root=_env_path(
                    "RL4RL_OPENEVOLVE_V21_RUNTIME",
                    Path("/private/tmp/rl4rl-c0c3-openevolve-v2-1"),
                ),
                campaign=_env_path(
                    "RL4RL_OPENEVOLVE_V21_CAMPAIGN",
                    REPO_ROOT
                    / "data/c0c3/controlled-openevolve-transformer-v2-1-mps-campaign",
                ),
                mode="individual-trajectories",
                blocks=(1, 2, 3),
            ),
        )
    if selected_profile != "primary":
        raise RuntimeError(f"unknown overnight profile: {selected_profile}")
    return (
        CampaignPlan(
            key="openevolve",
            runtime_root=_env_path(
                "RL4RL_OPENEVOLVE_RUNTIME",
                Path("/private/tmp/rl4rl-openevolve-resume"),
            ),
            campaign=_env_path(
                "RL4RL_OPENEVOLVE_CAMPAIGN",
                REPO_ROOT
                / "data/c0c3/workshop-pilot-parallel-adderboard-openevolve-campaign",
            ),
            mode="parallel-campaign",
        ),
        CampaignPlan(
            key="autoresearch-v1.5-6080",
            runtime_root=_env_path(
                "RL4RL_V15_RUNTIME",
                Path("/private/tmp/rl4rl-c0c3-autoresearch-v1_5"),
            ),
            campaign=_env_path(
                "RL4RL_V15_CAMPAIGN",
                Path("/private/tmp/rl4rl-v15-individual-campaign-live-20260822c"),
            ),
            mode="individual-trajectories",
            blocks=(1, 2),
        ),
        CampaignPlan(
            key="autoresearch-v1.5-1644",
            runtime_root=_env_path(
                "RL4RL_V15_1644_RUNTIME",
                Path("/private/tmp/rl4rl-c0c3-v15-codex1644"),
            ),
            campaign=_env_path(
                "RL4RL_V15_1644_CAMPAIGN",
                Path("/private/tmp/rl4rl-v15-codex1644-campaign-live-20260822a"),
            ),
            mode="individual-trajectories",
            blocks=(1,),
        ),
    )


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_schedule(campaign: Path) -> list[dict[str, Any]]:
    value = read_json(campaign / "schedule.json")
    if not isinstance(value, list):
        raise RuntimeError(f"campaign schedule is missing or invalid: {campaign}")
    return value


def expand_jobs(campaign_plans: tuple[CampaignPlan, ...] | None = None) -> list[Job]:
    expanded: list[Job] = []
    for plan in campaign_plans or plans():
        if plan.mode != "individual-trajectories":
            expanded.append(
                Job(
                    key=plan.key,
                    group=plan.key,
                    runtime_root=plan.runtime_root,
                    campaign=plan.campaign,
                    mode=plan.mode,
                    blocks=plan.blocks,
                    pause_after_proposals=plan.pause_after_proposals,
                )
            )
            continue
        for assignment in load_schedule(plan.campaign):
            block = int(assignment["block"])
            condition = str(assignment["condition"])
            if block not in plan.blocks or condition not in {"C0", "C1", "C2", "C3"}:
                continue
            run_id = str(assignment["run_id"])
            expanded.append(
                Job(
                    key=f"{plan.key}:b{block:02d}-{condition.lower()}",
                    group=plan.key,
                    runtime_root=plan.runtime_root,
                    campaign=plan.campaign,
                    mode=plan.mode,
                    run_id=run_id,
                    blocks=(block,),
                    pause_after_proposals=plan.pause_after_proposals,
                )
            )
    return expanded


def targeted_run_dirs(job: Job) -> list[Path]:
    assignments = load_schedule(job.campaign)
    selected: list[Path] = []
    for assignment in assignments:
        run_id = str(assignment["run_id"])
        if job.run_id is not None and run_id != job.run_id:
            continue
        if job.blocks and int(assignment["block"]) not in job.blocks:
            continue
        if job.mode == "independent-campaign" and str(assignment["condition"]) == "N0":
            continue
        selected.append(job.campaign / "runs" / run_id)
    return selected


def state_for(run_dir: Path) -> dict[str, Any]:
    value = read_json(run_dir / "state.json")
    if not isinstance(value, dict):
        raise RuntimeError(f"run state is missing or invalid: {run_dir}")
    return value


def progress_for(job: Job) -> dict[str, Any]:
    states = [state_for(run_dir) for run_dir in targeted_run_dirs(job)]
    proposals = [int(state.get("proposals_used", 0)) for state in states]
    tokens = [
        int(state.get("usage", {}).get("input_tokens", 0))
        + int(state.get("usage", {}).get("output_tokens", 0))
        for state in states
    ]
    best: list[int] = []
    for state in states:
        for candidate in state.get("candidates", {}).values():
            metrics = candidate.get("metrics", {})
            parameters = metrics.get("parameters")
            accuracy = metrics.get("accuracy", 0)
            if (
                isinstance(parameters, int)
                and isinstance(accuracy, int | float)
                and accuracy >= 0.99
            ):
                best.append(parameters)
    return {
        "runs": len(states),
        "completed_runs": sum(state.get("status") == "completed" for state in states),
        "active_opportunities": sum(
            state.get("active") is not None for state in states
        ),
        "min_proposals": min(proposals, default=0),
        "max_proposals": max(proposals, default=0),
        "total_tokens": sum(tokens),
        "lowest_parameters": min(best) if best else None,
    }


def job_completed(job: Job) -> bool:
    states = [state_for(run_dir) for run_dir in targeted_run_dirs(job)]
    return bool(states) and all(state.get("status") == "completed" for state in states)


def active_run_ids(job: Job) -> list[str]:
    return [
        run_dir.name
        for run_dir in targeted_run_dirs(job)
        if state_for(run_dir).get("active") is not None
    ]


def automatic_pause_reason(job: Job) -> str | None:
    """Arm a cooperative pause while the configured final proposal is active."""

    limit = job.pause_after_proposals
    if limit is None:
        return None
    for run_dir in targeted_run_dirs(job):
        state = state_for(run_dir)
        proposals_used = int(state.get("proposals_used", 0))
        active = state.get("active")
        active_index = (
            int(active.get("index", 0)) if isinstance(active, dict) else None
        )
        if proposals_used >= limit or (
            active_index is not None and active_index >= limit
        ):
            return f"automatic pause after proposal {limit}"
    return None


def cli_prefix(job: Job) -> list[str]:
    return [str(PYTHON_BIN), "-m", "experiments.c0c3_factorial.cli"]


def command_for(job: Job) -> list[str]:
    common = [
        "--campaign",
        str(job.campaign),
        "--python-bin",
        str(PYTHON_BIN),
        "--codex-binary",
        shutil.which("codex") or "codex",
    ]
    if job.mode == "parallel-campaign":
        return cli_prefix(job) + ["run-parallel-campaign", *common]
    if job.mode == "independent-campaign":
        return cli_prefix(job) + [
            "run-staged-independent-campaign",
            *common,
            "--block",
            str(job.blocks[0]),
            "--stage",
            "factorial",
        ]
    if job.mode == "individual-trajectories":
        state = state_for(job.campaign / "runs" / str(job.run_id))
        action = (
            "start-staged-trajectory"
            if int(state.get("proposals_used", 0)) == 0
            else "resume-staged-trajectory"
        )
        return cli_prefix(job) + [action, *common, "--run-id", str(job.run_id)]
    raise RuntimeError(f"unknown job mode: {job.mode}")


def recover_command(job: Job, run_id: str, reason: str) -> list[str]:
    return cli_prefix(job) + [
        "recover-active",
        "--campaign",
        str(job.campaign),
        "--run-id",
        run_id,
        "--reason",
        reason,
    ]


def pause_command(job: Job, reason: str) -> list[str]:
    if job.mode != "individual-trajectories" or job.run_id is None:
        raise RuntimeError("cooperative pause exists only for v1.5 trajectories")
    return cli_prefix(job) + [
        "pause-staged-trajectory",
        "--campaign",
        str(job.campaign),
        "--run-id",
        job.run_id,
        "--reason",
        reason,
    ]


RUNTIME_HASH_PROBE = r"""
import json
from pathlib import Path
import sys
from experiments.c0c3_factorial.cli import _load_campaign
from experiments.c0c3_factorial.artifacts import scientific_runtime_hash
campaign = Path(sys.argv[1])
repo_root = Path.cwd()
_spec, task, framework = _load_campaign(campaign)
print(scientific_runtime_hash(repo_root, task=task, framework=framework))
"""

ACCELERATOR_PROBE = r"""
import json
import sys
import torch
device = sys.argv[1]
available = (
    torch.backends.mps.is_available()
    if device == "mps"
    else torch.cuda.is_available()
    if device == "cuda"
    else True
)
result = {
    "device": device,
    "available": bool(available),
    "torch": torch.__version__,
}
if available and device in {"mps", "cuda"}:
    tensor = torch.ones(4, device=device)
    result["smoke_sum"] = float(tensor.sum().item())
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if available else 2)
"""


def runtime_hash(job: Job) -> str:
    result = subprocess.run(
        [str(PYTHON_BIN), "-c", RUNTIME_HASH_PROBE, str(job.campaign)],
        cwd=job.runtime_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"runtime hash probe failed for {job.group}: {result.stderr.strip()}"
        )
    return result.stdout.strip().splitlines()[-1]


def required_local_accelerator(campaign: Path) -> str | None:
    """Return a frozen local training device, excluding remote evaluators."""

    task = read_json(campaign / "inputs/task.json", {})
    if not isinstance(task, dict) or task.get("preferred_backend") != "local":
        return None
    command = task.get("evaluator_command")
    if not isinstance(command, list) or "--train-device" not in command:
        return None
    index = command.index("--train-device")
    if index + 1 >= len(command):
        return None
    device = str(command[index + 1])
    return device if device in {"mps", "cuda"} else None


def accelerator_error(job: Job) -> str | None:
    device = required_local_accelerator(job.campaign)
    if device is None:
        return None
    result = subprocess.run(
        [str(PYTHON_BIN), "-c", ACCELERATOR_PROBE, device],
        cwd=job.runtime_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    details = (result.stdout + result.stderr).strip()
    return (
        f"local {device} accelerator is unavailable to the supervisor process "
        f"for {job.group}: {details or 'probe returned no details'}. "
        "Launch from an ordinary terminal with accelerator access or use a "
        "separately calibrated remote-backend campaign."
    )


def preflight(jobs: list[Job], *, allow_active: bool) -> list[str]:
    errors: list[str] = []
    if not PYTHON_BIN.is_file():
        errors.append(f"Python environment is missing: {PYTHON_BIN}")
    for required in ("codex", "screen", "caffeinate"):
        if shutil.which(required) is None:
            errors.append(f"required executable is missing: {required}")
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    if not codex_home.is_dir() or not os.access(codex_home, os.W_OK):
        errors.append(f"Codex state directory is not writable: {codex_home}")

    checked: set[tuple[Path, Path]] = set()
    for job in jobs:
        if not job.runtime_root.is_dir():
            errors.append(
                f"runtime worktree is missing for {job.group}: {job.runtime_root}"
            )
            continue
        if not (job.campaign / "campaign.json").is_file():
            errors.append(f"campaign is missing for {job.group}: {job.campaign}")
            continue
        pair = (job.runtime_root, job.campaign)
        if pair not in checked:
            checked.add(pair)
            try:
                actual = runtime_hash(job)
                expected = str(
                    read_json(job.campaign / "campaign.json", {}).get(
                        "scientific_runtime_hash", ""
                    )
                )
                if actual != expected:
                    errors.append(
                        f"scientific runtime mismatch for {job.group}: "
                        f"expected {expected}, got {actual}"
                    )
            except RuntimeError as error:
                errors.append(str(error))
            hardware_error = accelerator_error(job)
            if hardware_error is not None:
                errors.append(hardware_error)
        if not allow_active:
            active = active_run_ids(job)
            if active:
                errors.append(
                    f"{job.key} has interrupted active opportunities: "
                    f"{', '.join(active)}"
                )
    return errors


def default_desired(jobs: list[Job]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now(),
        "jobs": {
            job.key: {"desired": "running", "reason": "overnight start"} for job in jobs
        },
    }


def load_desired(jobs: list[Job]) -> dict[str, Any]:
    desired = read_json(DESIRED_PATH)
    if not isinstance(desired, dict) or not isinstance(desired.get("jobs"), dict):
        desired = default_desired(jobs)
    for job in jobs:
        desired["jobs"].setdefault(
            job.key, {"desired": "running", "reason": "overnight start"}
        )
    return desired


def desired_for(job: Job) -> tuple[str, str]:
    desired = load_desired(expand_jobs()).get("jobs", {}).get(job.key, {})
    return str(desired.get("desired", "running")), str(desired.get("reason", ""))


def select_jobs(jobs: list[Job], targets: list[str]) -> list[Job]:
    if not targets or targets == ["all"]:
        return jobs
    selected = [job for job in jobs if job.key in targets or job.group in targets]
    unknown = sorted(
        set(targets) - {value for job in selected for value in (job.key, job.group)}
    )
    if unknown:
        raise SystemExit(f"unknown job/group: {', '.join(unknown)}")
    return selected


def set_desired(jobs: list[Job], targets: list[str], value: str, reason: str) -> None:
    chosen = select_jobs(jobs, targets)
    with file_lock(CONTROL_LOCK):
        desired = load_desired(jobs)
        for job in chosen:
            desired["jobs"][job.key] = {
                "desired": value,
                "reason": reason,
                "requested_at": utc_now(),
            }
        desired["updated_at"] = utc_now()
        atomic_json(DESIRED_PATH, desired)


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_group(
    process: subprocess.Popen[Any], *, grace_seconds: float = 45.0
) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGINT)
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.25)
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)


class Supervisor:
    def __init__(self, jobs: list[Job], *, recover_interrupted: bool):
        self.jobs = jobs
        self.recover_interrupted = recover_interrupted
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.processes: dict[str, subprocess.Popen[Any]] = {}
        self.runtime: dict[str, dict[str, Any]] = {
            job.key: {
                "actual": "starting",
                "child_pid": None,
                "restarts": 0,
                "last_exit_code": None,
                "last_error": None,
            }
            for job in jobs
        }

    def update(self, job: Job, **values: Any) -> None:
        with self.lock:
            self.runtime[job.key].update(values)

    def log(self, message: str) -> None:
        CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
        line = f"{utc_now()} {message}\n"
        with SUPERVISOR_LOG.open("a", encoding="utf-8") as handle:
            handle.write(line)
        print(line, end="", flush=True)

    def recover(self, job: Job, *, reason: str) -> None:
        active = active_run_ids(job)
        if not active:
            return
        if not self.recover_interrupted:
            raise RuntimeError(
                f"{job.key} has interrupted opportunities and recovery was not "
                "authorized"
            )
        for run_id in active:
            self.log(f"recovering {job.key} run={run_id}: {reason}")
            result = subprocess.run(
                recover_command(job, run_id, reason),
                cwd=job.runtime_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            with (CONTROL_ROOT / "recovery.log").open("a", encoding="utf-8") as handle:
                handle.write(result.stdout)
            if result.returncode != 0:
                raise RuntimeError(
                    f"recovery failed for {job.key}/{run_id}: {result.stdout.strip()}"
                )

    def request_cooperative_pause(self, job: Job, reason: str) -> None:
        result = subprocess.run(
            pause_command(job, reason),
            cwd=job.runtime_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if (
            result.returncode != 0
            and "completed trajectories cannot be paused" not in result.stdout
        ):
            raise RuntimeError(
                f"pause request failed for {job.key}: {result.stdout.strip()}"
            )

    def worker(self, job: Job) -> None:
        backoff = 15.0
        last_progress = -1
        pause_sent = False
        log_path = CONTROL_ROOT / "logs" / f"{job.key.replace(':', '_')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        while not self.stop_event.is_set():
            desired, reason = desired_for(job)
            with self.lock:
                process = self.processes.get(job.key)

            if desired == "running":
                automatic_reason = automatic_pause_reason(job)
                if automatic_reason is not None:
                    set_desired(
                        self.jobs,
                        [job.key],
                        "paused",
                        automatic_reason,
                    )
                    desired, reason = "paused", automatic_reason
                    self.log(f"armed {job.key}: {automatic_reason}")

            if desired != "running":
                if process is not None and process.poll() is None:
                    self.update(
                        job, actual="pausing" if desired == "paused" else "stopping"
                    )
                    if job.mode == "individual-trajectories":
                        if not pause_sent:
                            try:
                                self.request_cooperative_pause(job, reason or desired)
                                pause_sent = True
                            except RuntimeError as error:
                                self.update(job, last_error=str(error))
                    else:
                        terminate_group(process)
                if process is None or process.poll() is not None:
                    with self.lock:
                        self.processes.pop(job.key, None)
                    self.update(
                        job,
                        actual="paused" if desired == "paused" else "stopped",
                        child_pid=None,
                    )
                time.sleep(POLL_SECONDS)
                continue

            pause_sent = False
            if job_completed(job):
                self.update(job, actual="completed", child_pid=None)
                return

            if process is not None and process.poll() is None:
                self.update(job, actual="running", child_pid=process.pid)
                time.sleep(POLL_SECONDS)
                continue

            if process is not None:
                exit_code = process.poll()
                with self.lock:
                    self.processes.pop(job.key, None)
                self.update(job, child_pid=None, last_exit_code=exit_code)
                current_progress = progress_for(job)["min_proposals"]
                if current_progress > last_progress:
                    backoff = 15.0
                else:
                    backoff = min(MAX_BACKOFF_SECONDS, max(30.0, backoff * 2.0))
                last_progress = current_progress
                try:
                    self.recover(
                        job,
                        reason=(
                            "overnight supervisor recovered an opportunity after "
                            f"controller exit code {exit_code}"
                        ),
                    )
                except RuntimeError as error:
                    self.update(job, actual="recovery-error", last_error=str(error))
                    self.log(str(error))
                    time.sleep(min(backoff, 300.0))
                    continue
                self.update(job, actual="backing-off", backoff_seconds=backoff)
                time.sleep(backoff)
                continue

            try:
                self.recover(
                    job,
                    reason="overnight supervisor startup after confirmed absent writer",
                )
                command = command_for(job)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n{utc_now()} START {json.dumps(command)}\n")
                    handle.flush()
                    process = subprocess.Popen(
                        command,
                        cwd=job.runtime_root,
                        stdin=subprocess.DEVNULL,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                with self.lock:
                    self.processes[job.key] = process
                    self.runtime[job.key]["restarts"] += 1
                self.update(
                    job, actual="running", child_pid=process.pid, last_error=None
                )
                self.log(f"started {job.key} pid={process.pid}")
            except Exception as error:  # noqa: BLE001 - daemon must remain observable
                self.update(
                    job,
                    actual="launch-error",
                    last_error=f"{type(error).__name__}: {error}",
                )
                self.log(f"launch error for {job.key}: {type(error).__name__}: {error}")
                time.sleep(backoff)
                backoff = min(MAX_BACKOFF_SECONDS, max(30.0, backoff * 2.0))

    def write_status(self) -> None:
        desired = load_desired(self.jobs)
        with self.lock:
            runtime = json.loads(json.dumps(self.runtime))
        for job in self.jobs:
            runtime[job.key]["desired"] = desired["jobs"][job.key]["desired"]
            try:
                runtime[job.key]["progress"] = progress_for(job)
            except Exception as error:  # noqa: BLE001 - status remains available
                runtime[job.key]["progress_error"] = f"{type(error).__name__}: {error}"
        atomic_json(
            STATUS_PATH,
            {
                "schema_version": SCHEMA_VERSION,
                "supervisor_pid": os.getpid(),
                "heartbeat_at": utc_now(),
                "jobs": runtime,
            },
        )

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            processes = list(self.processes.values())
        for process in processes:
            terminate_group(process)

    def run(self) -> int:
        threads = [
            threading.Thread(target=self.worker, args=(job,), name=job.key, daemon=True)
            for job in self.jobs
        ]
        for thread in threads:
            thread.start()
        try:
            while not self.stop_event.is_set():
                self.write_status()
                if all(not thread.is_alive() for thread in threads):
                    break
                time.sleep(POLL_SECONDS)
        finally:
            self.stop()
            for thread in threads:
                thread.join(timeout=60)
            self.write_status()
        return 0


def screen_running() -> bool:
    result = subprocess.run(
        [shutil.which("screen") or "/usr/bin/screen", "-ls"],
        text=True,
        capture_output=True,
        check=False,
    )
    return SCREEN_SESSION in (result.stdout + result.stderr)


def command_check(args: argparse.Namespace) -> int:
    jobs = expand_jobs()
    errors = preflight(jobs, allow_active=args.allow_active)
    for job in jobs:
        progress = progress_for(job)
        print(
            f"{job.key}\truns={progress['runs']}\t"
            f"proposals={progress['min_proposals']}-{progress['max_proposals']}\t"
            f"active={progress['active_opportunities']}\t"
            f"tokens={progress['total_tokens']}"
        )
    if errors:
        print("\nPreflight failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("\nPreflight passed.")
    return 0


def command_start(args: argparse.Namespace) -> int:
    CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = expand_jobs()
    if screen_running():
        raise SystemExit(f"screen session {SCREEN_SESSION!r} is already running")
    errors = preflight(jobs, allow_active=args.recover_interrupted)
    if errors:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(errors))
    if args.all_running or not DESIRED_PATH.exists():
        atomic_json(DESIRED_PATH, default_desired(jobs))
    screen = shutil.which("screen") or "/usr/bin/screen"
    caffeinate = shutil.which("caffeinate") or "/usr/bin/caffeinate"
    command = [
        screen,
        "-L",
        "-dmS",
        SCREEN_SESSION,
        caffeinate,
        "-dimsu",
        str(PYTHON_BIN),
        str(Path(__file__).resolve()),
        "daemon",
    ]
    if args.recover_interrupted:
        command.append("--recover-interrupted")
    result = subprocess.run(command, cwd=CONTROL_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"screen failed to start (exit {result.returncode})")
    time.sleep(1.0)
    if not screen_running():
        raise SystemExit(
            f"screen session exited; inspect {CONTROL_ROOT / 'screenlog.0'}"
        )
    print(f"started detached session {SCREEN_SESSION}")
    print(f"status: {PYTHON_BIN} {Path(__file__).resolve()} status")
    print(f"logs:   {CONTROL_ROOT}")
    return 0


def command_daemon(args: argparse.Namespace) -> int:
    CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
    with SUPERVISOR_LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(
                "another overnight supervisor already owns the lock"
            ) from error
        jobs = expand_jobs()
        errors = preflight(jobs, allow_active=args.recover_interrupted)
        if errors:
            raise SystemExit("daemon preflight failed:\n- " + "\n- ".join(errors))
        supervisor = Supervisor(jobs, recover_interrupted=args.recover_interrupted)

        def stop_handler(_signum: int, _frame: Any) -> None:
            supervisor.stop()

        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        return supervisor.run()


def command_control(args: argparse.Namespace) -> int:
    jobs = expand_jobs()
    reason = args.reason or f"operator requested {args.command}"
    value = "running" if args.command == "resume" else args.command + "d"
    if args.command == "pause":
        value = "paused"
    elif args.command == "stop":
        value = "stopped"
    set_desired(jobs, args.targets, value, reason)
    selected = select_jobs(jobs, args.targets)
    print(f"{value}: " + ", ".join(job.key for job in selected))
    return 0


def command_status(_args: argparse.Namespace) -> int:
    jobs = expand_jobs()
    stored = read_json(STATUS_PATH, {})
    heartbeat = stored.get("heartbeat_at") if isinstance(stored, dict) else None
    supervisor_pid = stored.get("supervisor_pid") if isinstance(stored, dict) else None
    live = screen_running() and process_alive(supervisor_pid)
    print(
        f"supervisor={'running' if live else 'not-running'} "
        f"pid={supervisor_pid or '-'} heartbeat={heartbeat or '-'}"
    )
    print("job\tdesired\tactual\tpid\tproposals\ttokens\tlowest_params\tactive")
    desired = load_desired(jobs)
    runtime = stored.get("jobs", {}) if isinstance(stored, dict) else {}
    for job in jobs:
        progress = progress_for(job)
        item = runtime.get(job.key, {})
        print(
            "\t".join(
                (
                    job.key,
                    str(desired["jobs"][job.key]["desired"]),
                    str(item.get("actual", "not-running")),
                    str(item.get("child_pid") or "-"),
                    f"{progress['min_proposals']}-{progress['max_proposals']}",
                    str(progress["total_tokens"]),
                    str(
                        progress["lowest_parameters"]
                        if progress["lowest_parameters"] is not None
                        else "-"
                    ),
                    str(progress["active_opportunities"]),
                )
            )
        )
    return 0


def command_shutdown(_args: argparse.Namespace) -> int:
    stored = read_json(STATUS_PATH, {})
    pid = stored.get("supervisor_pid") if isinstance(stored, dict) else None
    if process_alive(pid):
        os.kill(int(pid), signal.SIGTERM)
        print(f"shutdown requested for supervisor pid {pid}")
    else:
        print("supervisor is not running")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="verify runtimes without starting")
    check.add_argument("--allow-active", action="store_true")
    check.set_defaults(handler=command_check)

    start = subparsers.add_parser("start", help="start detached overnight supervisor")
    start.add_argument("--recover-interrupted", action="store_true")
    start.add_argument("--all-running", action="store_true")
    start.set_defaults(handler=command_start)

    daemon = subparsers.add_parser("daemon", help=argparse.SUPPRESS)
    daemon.add_argument("--recover-interrupted", action="store_true")
    daemon.set_defaults(handler=command_daemon)

    for name in ("pause", "resume", "stop"):
        control = subparsers.add_parser(name)
        control.add_argument("targets", nargs="*", help="job/group names; default all")
        control.add_argument("--reason")
        control.set_defaults(handler=command_control)

    status = subparsers.add_parser("status")
    status.set_defaults(handler=command_status)

    shutdown = subparsers.add_parser("shutdown")
    shutdown.set_defaults(handler=command_shutdown)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

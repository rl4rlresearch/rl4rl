#!/usr/bin/env python3
"""Generic cooperative pause/resume controls for dashboard campaigns.

This module is operational infrastructure, not part of a campaign's scientific
runtime.  It adapts the two durable controller families used by the repository:

* semantic campaigns drain through ``semantic-control.json``;
* per-trajectory overnight supervisors drain through their ``desired.json``.

No action sends a signal to a subject or evaluator process.  A pause changes a
desired state and lets the owning controller finish already-started work before
exiting.  Resume restores the desired state and starts an absent durable
supervisor when its launch metadata is available.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROL_GLOB = "overnight-control*"
SEMANTIC_SUPERVISOR_METADATA = Path("semantic-supervisor.json")
LIFECYCLE_HISTORY = Path("dashboard-lifecycle.jsonl")


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def process_alive(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def screen_sessions() -> set[str]:
    screen = shutil.which("screen") or "/usr/bin/screen"
    try:
        result = subprocess.run(
            (screen, "-ls"), capture_output=True, text=True, check=False, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    sessions: set[str] = set()
    for line in (result.stdout + result.stderr).splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        _pid, separator, name = fields[0].partition(".")
        if separator and name:
            sessions.add(name)
    return sessions


def semantic_screen_session(campaign: Path) -> str:
    digest = hashlib.sha256(str(campaign.resolve()).encode()).hexdigest()[:10]
    return f"rl4rl-semantic-{digest}"


def _normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _campaign_states(campaign: Path) -> list[dict[str, Any]]:
    return [
        value
        for value in (
            read_json(path / "state.json", {})
            for path in (campaign / "runs").glob("*")
            if path.is_dir()
        )
        if isinstance(value, dict)
    ]


RESEARCH_ARCHITECTURE_LABELS = {
    "karpathy_autoresearch": "Autoresearch",
    "openevolve": "Greedy OpenEvolve",
    "native_openevolve": "Native OpenEvolve",
}


def campaign_run_identity(campaign: Path) -> dict[str, str]:
    task = read_json(campaign / "inputs/task.json", {})
    task = task if isinstance(task, dict) else {}
    framework = read_json(campaign / "inputs/framework.json", {})
    framework = framework if isinstance(framework, dict) else {}
    protocol = read_json(campaign / "inputs/protocol.json", {})
    protocol = protocol if isinstance(protocol, dict) else {}
    manifest = read_json(campaign / "campaign.json", {})
    manifest = manifest if isinstance(manifest, dict) else {}

    architecture = str(framework.get("framework_id") or "unknown")
    protocol_version = str(protocol.get("protocol_version") or "unknown")
    task_id = str(task.get("task_id") or "unknown")
    return {
        "task_id": task_id,
        "task_display_name": str(task.get("display_name") or task_id),
        "research_architecture": architecture,
        "research_architecture_label": RESEARCH_ARCHITECTURE_LABELS.get(
            architecture, architecture
        ),
        "protocol_version": protocol_version,
        "protocol_study_id": str(
            protocol.get("study_id") or manifest.get("study_id") or "unknown"
        ),
    }


def _semantic_binding(
    campaign_id: str,
    campaign: Path,
    *,
    sessions: set[str],
) -> dict[str, Any] | None:
    control_path = campaign / "semantic-control.json"
    if not control_path.is_file():
        return None
    control = read_json(control_path, {})
    desired = str(control.get("desired", "stopped"))
    states = _campaign_states(campaign)
    active = sum(isinstance(state.get("active"), dict) for state in states)
    completed = bool(states) and all(
        state.get("status") == "completed" for state in states
    )
    session = semantic_screen_session(campaign)
    supervisor_live = session in sessions
    if completed:
        actual = "completed"
    elif desired == "paused":
        actual = "pausing" if active or supervisor_live else "paused"
    elif desired == "running":
        actual = "running" if supervisor_live else "supervisor-offline"
    else:
        actual = "stopping" if active or supervisor_live else "stopped"
    return {
        "campaign_id": campaign_id,
        "backend": "semantic_campaign",
        "desired": desired,
        "actual": actual,
        "active_opportunities": active,
        "completed": completed,
        "supervisor_live": supervisor_live,
        "screen_session": session,
        "control_paths": [str(control_path)],
        "bindings": [],
        "supported": True,
        "safe_boundary": "after already-started proposals finish",
    }


def _overnight_group_matches(
    campaign_id: str,
    campaign: Path,
    group: str,
    declared_campaigns: Iterable[object],
) -> bool:
    target = str(campaign.resolve())
    if any(
        str(Path(str(value)).expanduser().resolve()) == target
        for value in declared_campaigns
        if value
    ):
        return True
    group_key = _normalized(group)
    identifiers = {
        _normalized(campaign_id),
        _normalized(campaign.name.removesuffix("-campaign")),
    }
    if group_key in identifiers:
        return True
    # Extension supervisors can own later blocks of the same campaign.
    return (
        group_key.endswith("extension")
        and group_key.removesuffix("extension") in identifiers
    )


def _overnight_bindings(
    campaign_id: str,
    campaign: Path,
    *,
    control_parent: Path,
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    if not control_parent.is_dir():
        return bindings
    for root in sorted(control_parent.glob(DEFAULT_CONTROL_GLOB)):
        if not root.is_dir():
            continue
        desired = read_json(root / "desired.json", {})
        status = read_json(root / "status.json", {})
        metadata = read_json(root / "supervisor-metadata.json", {})
        desired_jobs = desired.get("jobs", {}) if isinstance(desired, dict) else {}
        status_jobs = status.get("jobs", {}) if isinstance(status, dict) else {}
        metadata_jobs = metadata.get("jobs", {}) if isinstance(metadata, dict) else {}
        if not isinstance(desired_jobs, dict) or not isinstance(status_jobs, dict):
            continue
        groups: dict[str, list[str]] = {}
        for job_key in set(desired_jobs) | set(status_jobs) | set(metadata_jobs):
            group = str(job_key).split(":", 1)[0]
            declared = []
            for source in (status_jobs, metadata_jobs):
                item = source.get(job_key, {})
                if isinstance(item, dict):
                    declared.append(item.get("campaign"))
            if _overnight_group_matches(campaign_id, campaign, group, declared):
                groups.setdefault(group, []).append(str(job_key))
        if not groups:
            continue
        profile = str(
            metadata.get("profile")
            or root.name.removeprefix("overnight-control-")
            or "primary"
        )
        if root.name == "overnight-control":
            profile = "primary"
        bindings.append(
            {
                "control_root": str(root.resolve()),
                "desired_path": str((root / "desired.json").resolve()),
                "status_path": str((root / "status.json").resolve()),
                "profile": profile,
                "groups": sorted(groups),
                "job_keys": sorted({key for keys in groups.values() for key in keys}),
                "supervisor_pid": status.get("supervisor_pid"),
                "supervisor_live": process_alive(status.get("supervisor_pid")),
                "screen_session": metadata.get("screen_session"),
                "status_jobs": status_jobs,
                "desired_jobs": desired_jobs,
            }
        )
    return bindings


def _overnight_binding_payload(
    campaign_id: str,
    bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    desired_values: list[str] = []
    actual_values: list[str] = []
    active = 0
    for binding in bindings:
        for key in binding["job_keys"]:
            desired_item = binding["desired_jobs"].get(key, {})
            status_item = binding["status_jobs"].get(key, {})
            if isinstance(desired_item, dict):
                desired_values.append(str(desired_item.get("desired", "running")))
            if isinstance(status_item, dict):
                actual_values.append(str(status_item.get("actual", "not-running")))
                progress = status_item.get("progress", {})
                if isinstance(progress, dict):
                    active += int(progress.get("active_opportunities", 0) or 0)
    completed = bool(actual_values) and all(
        value == "completed" for value in actual_values
    )
    desired = (
        desired_values[0]
        if desired_values and len(set(desired_values)) == 1
        else "mixed"
    )
    supervisor_live = any(bool(binding["supervisor_live"]) for binding in bindings)
    if completed:
        actual = "completed"
    elif desired == "paused":
        paused_states = {"paused", "completed", "stopped", "not-running"}
        actual = (
            "paused"
            if actual_values and all(value in paused_states for value in actual_values)
            else "pausing"
        )
    elif desired == "running":
        actual = "running" if supervisor_live else "supervisor-offline"
    elif desired == "stopped":
        actual = "stopped" if not supervisor_live else "stopping"
    else:
        actual = "mixed"
    return {
        "campaign_id": campaign_id,
        "backend": "overnight_supervisor",
        "desired": desired,
        "actual": actual,
        "active_opportunities": active,
        "completed": completed,
        "supervisor_live": supervisor_live,
        "control_paths": [binding["desired_path"] for binding in bindings],
        "bindings": [
            {
                key: value
                for key, value in binding.items()
                if key not in {"status_jobs", "desired_jobs"}
            }
            for binding in bindings
        ],
        "supported": True,
        "safe_boundary": "after each trajectory's already-started proposal finishes",
    }


def campaign_lifecycle_payload(
    campaigns: Mapping[str, Path],
    dashboard_snapshot: Mapping[str, Any] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    sessions: set[str] | None = None,
) -> dict[str, Any]:
    """Describe safe lifecycle controls for every existing campaign."""

    sessions = screen_sessions() if sessions is None else sessions
    snapshot_campaigns = (dashboard_snapshot or {}).get("campaigns", {})
    rows: list[dict[str, Any]] = []
    for campaign_id, campaign_value in sorted(campaigns.items()):
        campaign = Path(campaign_value).resolve()
        if not (campaign / "campaign.json").is_file():
            continue
        semantic = _semantic_binding(campaign_id, campaign, sessions=sessions)
        if semantic is not None:
            row = semantic
        else:
            bindings = _overnight_bindings(
                campaign_id,
                campaign,
                control_parent=repo_root / "data/c0c3",
            )
            if bindings:
                row = _overnight_binding_payload(campaign_id, bindings)
            else:
                states = _campaign_states(campaign)
                completed = bool(states) and all(
                    state.get("status") == "completed" for state in states
                )
                row = {
                    "campaign_id": campaign_id,
                    "backend": "unmanaged",
                    "desired": "completed" if completed else "unknown",
                    "actual": "completed" if completed else "not-supervised",
                    "active_opportunities": sum(
                        isinstance(state.get("active"), dict) for state in states
                    ),
                    "completed": completed,
                    "supervisor_live": False,
                    "control_paths": [],
                    "bindings": [],
                    "supported": completed,
                    "safe_boundary": None,
                }
        dashboard_row = snapshot_campaigns.get(campaign_id, {})
        manifest = read_json(campaign / "campaign.json", {})
        identity = campaign_run_identity(campaign)
        label = dashboard_row.get("label") or dashboard_row.get("task_display_name")
        if not label:
            label = identity["task_display_name"]
        if label == "unknown":
            label = manifest.get("study_id") or campaign.name
        row.update(
            {
                "id": campaign_id,
                "label": str(label),
                "path": str(campaign),
                **identity,
                "can_pause": bool(
                    row["supported"]
                    and not row["completed"]
                    and row["desired"] != "paused"
                ),
                "can_resume": bool(
                    row["supported"]
                    and not row["completed"]
                    and row["desired"] != "running"
                ),
            }
        )
        rows.append(row)
    return {
        "available": bool(rows),
        "campaigns": rows,
        "semantics": "cooperative_safe_boundary_no_active_work_interruption",
        "future_discovery": (
            "campaigns with standard semantic or overnight supervisor metadata "
            "are automatic"
        ),
    }


def _set_overnight_desired(
    binding: Mapping[str, Any], *, desired: str, reason: str
) -> dict[str, Any]:
    root = Path(str(binding["control_root"]))
    desired_path = Path(str(binding["desired_path"]))
    lock_path = root / "control.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        payload = read_json(desired_path, {})
        jobs = payload.get("jobs", {}) if isinstance(payload, dict) else {}
        if not isinstance(jobs, dict):
            raise ValueError(f"invalid overnight desired registry: {desired_path}")
        changed = []
        for key in binding["job_keys"]:
            if key not in jobs:
                continue
            jobs[key] = {
                "desired": desired,
                "reason": reason,
                "requested_at": utc_now(),
            }
            changed.append(key)
        if not changed:
            raise ValueError("campaign has no registered jobs in its supervisor")
        payload.update({"schema_version": "1.0", "updated_at": utc_now(), "jobs": jobs})
        atomic_json(desired_path, payload)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {"control_root": str(root), "jobs": changed}


def _start_overnight_supervisor(
    binding: Mapping[str, Any],
    *,
    repo_root: Path,
    run_command: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["RL4RL_OVERNIGHT_PROFILE"] = str(binding["profile"])
    environment["RL4RL_OVERNIGHT_CONTROL_ROOT"] = str(binding["control_root"])
    if binding.get("screen_session"):
        environment["RL4RL_OVERNIGHT_SCREEN_SESSION"] = str(binding["screen_session"])
    command = [
        str(repo_root / "architecture_discovery/.venv/bin/python"),
        str(repo_root / "experiments/c0c3_overnight.py"),
        "start",
        "--recover-interrupted",
    ]
    result = run_command(
        command,
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if (
        result.returncode != 0
        and "already running" not in result.stdout + result.stderr
    ):
        raise RuntimeError(
            "overnight supervisor failed to start: "
            + (result.stdout + result.stderr).strip()
        )
    return {"profile": binding["profile"], "output": result.stdout.strip()}


def _semantic_launch_metadata(campaign: Path, repo_root: Path) -> dict[str, Any]:
    stored = read_json(campaign / SEMANTIC_SUPERVISOR_METADATA, {})
    if not isinstance(stored, dict):
        stored = {}
    return {
        "runtime_root": str(Path(stored.get("runtime_root", repo_root)).resolve()),
        "scientific_repo_root": str(
            Path(stored.get("scientific_repo_root", repo_root)).resolve()
        ),
        "python_bin": str(
            Path(
                stored.get(
                    "python_bin",
                    repo_root / "architecture_discovery/.venv/bin/python",
                )
            ).absolute()
        ),
        "max_workers": int(stored.get("max_workers", 0)),
        "fashion_data_root": stored.get("fashion_data_root"),
    }


def _semantic_resume_command(campaign: Path, metadata: Mapping[str, Any]) -> list[str]:
    command = [
        str(metadata["python_bin"]),
        str(
            Path(str(metadata["runtime_root"]))
            / "experiments/semantic_intervention_overnight.py"
        ),
        "resume",
        "--campaign",
        str(campaign),
        "--runtime-root",
        str(metadata["runtime_root"]),
        "--scientific-repo-root",
        str(metadata["scientific_repo_root"]),
        "--python-bin",
        str(metadata["python_bin"]),
        "--max-workers",
        str(metadata["max_workers"]),
        "--reason",
        "controller dashboard resumed campaign",
    ]
    if metadata.get("fashion_data_root"):
        command.extend(("--fashion-data-root", str(metadata["fashion_data_root"])))
    return command


def _resume_semantic_supervisor(
    campaign: Path,
    *,
    repo_root: Path,
    was_desired: str,
    sessions: set[str],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    spawn_command: Callable[..., subprocess.Popen[Any]],
) -> dict[str, Any]:
    metadata = _semantic_launch_metadata(campaign, repo_root)
    session = semantic_screen_session(campaign)
    if session not in sessions:
        command = _semantic_resume_command(campaign, metadata)
        result = run_command(
            command,
            cwd=metadata["runtime_root"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "semantic supervisor failed to resume: "
                + (result.stdout + result.stderr).strip()
            )
        return {"mode": "started", "output": result.stdout.strip()}
    if was_desired not in {"paused", "stopped"}:
        return {"mode": "already-running"}

    # The semantic runner latches a pause once observed and then drains.  A
    # detached helper waits for that screen to exit and starts its successor,
    # avoiding a race when Resume is clicked during the drain.
    command = [
        str(metadata["python_bin"]),
        str(
            Path(str(metadata["runtime_root"]))
            / "experiments/semantic_intervention_overnight.py"
        ),
        "resume-after-drain",
        "--campaign",
        str(campaign),
        "--runtime-root",
        str(metadata["runtime_root"]),
        "--scientific-repo-root",
        str(metadata["scientific_repo_root"]),
        "--python-bin",
        str(metadata["python_bin"]),
        "--max-workers",
        str(metadata["max_workers"]),
        "--reason",
        "controller dashboard resumed campaign after cooperative drain",
    ]
    if metadata.get("fashion_data_root"):
        command.extend(("--fashion-data-root", str(metadata["fashion_data_root"])))
    log_path = campaign / "semantic-dashboard-resume.log"
    log_handle = log_path.open("a", encoding="utf-8")
    try:
        process = spawn_command(
            command,
            cwd=metadata["runtime_root"],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    return {"mode": "resume-after-drain", "pid": process.pid, "log": str(log_path)}


def set_campaign_lifecycle(
    campaigns: Mapping[str, Path],
    *,
    campaign_id: str,
    desired: str,
    reason: str,
    repo_root: Path = REPO_ROOT,
    sessions: set[str] | None = None,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    spawn_command: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> dict[str, Any]:
    """Request a cooperative campaign pause or a durable resume."""

    if desired not in {"paused", "running"}:
        raise ValueError("campaign lifecycle desired state must be paused or running")
    if campaign_id not in campaigns:
        raise ValueError("unknown campaign")
    if not reason.strip():
        raise ValueError("campaign lifecycle reason cannot be blank")
    campaign = Path(campaigns[campaign_id]).resolve()
    if not (campaign / "campaign.json").is_file():
        raise ValueError("campaign does not exist")
    sessions = screen_sessions() if sessions is None else sessions

    semantic = _semantic_binding(campaign_id, campaign, sessions=sessions)
    actions: list[dict[str, Any]] = []
    if semantic is not None:
        previous = semantic["desired"]
        from experiments.c0c3_factorial.semantic_interventions import (
            set_semantic_control,
        )

        receipt = set_semantic_control(campaign, desired=desired, reason=reason)
        actions.append({"backend": "semantic_campaign", "receipt": receipt})
        if desired == "running" and not semantic["completed"]:
            actions.append(
                _resume_semantic_supervisor(
                    campaign,
                    repo_root=repo_root,
                    was_desired=previous,
                    sessions=sessions,
                    run_command=run_command,
                    spawn_command=spawn_command,
                )
            )
    else:
        bindings = _overnight_bindings(
            campaign_id,
            campaign,
            control_parent=repo_root / "data/c0c3",
        )
        if not bindings:
            raise ValueError(
                "campaign has no durable lifecycle supervisor registration"
            )
        for binding in bindings:
            actions.append(
                _set_overnight_desired(binding, desired=desired, reason=reason)
            )
        if desired == "running":
            started_roots: set[str] = set()
            for binding in bindings:
                root = str(binding["control_root"])
                if binding["supervisor_live"] or root in started_roots:
                    continue
                actions.append(
                    _start_overnight_supervisor(
                        binding, repo_root=repo_root, run_command=run_command
                    )
                )
                started_roots.add(root)

    record = {
        "schema_version": "1.0",
        "event": "dashboard_campaign_lifecycle_requested",
        "timestamp": utc_now(),
        "campaign_id": campaign_id,
        "campaign": str(campaign),
        "desired": desired,
        "reason": reason.strip(),
        "actions": actions,
        "semantics": "cooperative_safe_boundary_no_active_work_interruption",
    }
    append_jsonl(repo_root / "data" / LIFECYCLE_HISTORY, record)
    return record


def discover_campaigns(
    configured: Mapping[str, Path],
    *,
    roots: Iterable[Path],
) -> dict[str, Path]:
    """Add standard campaign directories using stable collision-safe IDs."""

    result = {key: Path(value).resolve() for key, value in configured.items()}
    known = {path for path in result.values()}
    for parent in roots:
        if not parent.is_dir():
            continue
        for candidate in sorted(path for path in parent.iterdir() if path.is_dir()):
            resolved = candidate.resolve()
            if resolved in known or not (resolved / "campaign.json").is_file():
                continue
            slug = re.sub(r"[^a-z0-9]+", "_", candidate.name.lower()).strip("_")
            key = f"discovered_{slug}"
            if key in result and result[key] != resolved:
                digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:8]
                key = f"{key}_{digest}"
            result[key] = resolved
            known.add(resolved)
    return result


if __name__ == "__main__":
    raise SystemExit("This module is used by the local controller dashboard.")

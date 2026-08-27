"""Crash-releasing host-wide opportunity-worker scheduler.

The scheduler is deliberately independent of campaign orchestration. A slot
is one actively executing subject-agent call, and every campaign using the
current runtime competes for the same host-wide pool. Evaluators have their own
separate host and task pools. File locks make leases self-releasing after a
process crash while leaving campaigns free to advance independently.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

SHARED_AGENT_WORKER_CAPACITY = 30
SHARED_AGENT_WORKER_ROOT_ENV = "RL4RL_SHARED_AGENT_WORKER_ROOT"


class WorkerQueueCancelled(RuntimeError):
    """A not-yet-started opportunity was cancelled while waiting for a slot."""


def shared_agent_worker_root() -> Path:
    """Return one stable lock root shared by campaign runtimes on this host."""

    configured = os.environ.get(SHARED_AGENT_WORKER_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    stable_tmp = Path("/private/tmp")
    base = stable_tmp if stable_tmp.is_dir() else Path(tempfile.gettempdir())
    return base / f"rl4rl-c0c3-agent-workers-{os.getuid()}-v1"


@dataclass
class AgentWorkerLease:
    """One owned worker slot; callers must release or transfer ownership."""

    handle: TextIO
    path: Path
    index: int
    acquired_at_unix: float
    waited_seconds: float


def release_agent_worker_slot(lease: AgentWorkerLease | None) -> None:
    if lease is None or lease.handle.closed:
        return
    fcntl.flock(lease.handle.fileno(), fcntl.LOCK_UN)
    lease.handle.close()


def _ensure_scheduler(root: Path, capacity: int) -> None:
    if capacity < 1:
        raise ValueError("shared agent worker capacity must be positive")
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "scheduler.json"
    lock_path = root / "scheduler-config.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if config_path.is_file():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            configured_capacity = int(payload.get("capacity", 0))
            if configured_capacity != capacity:
                raise RuntimeError(
                    "shared agent worker scheduler capacity mismatch: "
                    f"requested {capacity}, found {configured_capacity}"
                )
        else:
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "capacity": capacity,
                        "scheduler": "crash_releasing_host_file_locks_v1",
                        "synchronization_barrier": False,
                        "scope": "all_campaigns",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _first_slot(worker_id: str, capacity: int) -> int:
    return (
        int.from_bytes(hashlib.sha256(worker_id.encode()).digest()[:8], "little")
        % capacity
    )


def try_acquire_agent_worker_slot(
    *,
    worker_id: str,
    metadata: dict[str, object] | None = None,
    root: Path | None = None,
    capacity: int = SHARED_AGENT_WORKER_CAPACITY,
    requested_at_unix: float | None = None,
) -> AgentWorkerLease | None:
    """Try once to claim any host slot without starting a scientific attempt."""

    selected_root = (root or shared_agent_worker_root()).resolve()
    _ensure_scheduler(selected_root, capacity)
    requested_at = requested_at_unix or time.time()
    first = _first_slot(worker_id, capacity)
    for offset in range(capacity):
        index = (first + offset) % capacity
        path = selected_root / f"slot-{index:02d}.lock"
        handle = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        acquired_at = time.time()
        holder = {
            "schema_version": "1.0",
            "pid": os.getpid(),
            "worker_id": worker_id,
            "requested_at_unix": requested_at,
            "acquired_at_unix": acquired_at,
            "waited_seconds": max(0.0, acquired_at - requested_at),
            **(metadata or {}),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(holder, handle, sort_keys=True)
        handle.flush()
        return AgentWorkerLease(
            handle=handle,
            path=path,
            index=index,
            acquired_at_unix=acquired_at,
            waited_seconds=float(holder["waited_seconds"]),
        )
    return None


def acquire_agent_worker_slot(
    *,
    worker_id: str,
    metadata: dict[str, object] | None = None,
    root: Path | None = None,
    capacity: int = SHARED_AGENT_WORKER_CAPACITY,
    cancel_path: Path | None = None,
    poll_seconds: float = 0.25,
) -> AgentWorkerLease:
    """Wait for a slot, without consuming an opportunity while queued.

    Individually controlled trajectories pass their cooperative-pause marker
    as ``cancel_path``.  A pause can therefore cancel queued, scientifically
    unstarted work immediately; already-acquired work still drains normally.
    """

    if poll_seconds <= 0:
        raise ValueError("worker slot polling interval must be positive")
    requested_at = time.time()
    while True:
        if cancel_path is not None and cancel_path.exists():
            raise WorkerQueueCancelled(
                f"worker queue cancelled before opportunity start: {cancel_path}"
            )
        lease = try_acquire_agent_worker_slot(
            worker_id=worker_id,
            metadata=metadata,
            root=root,
            capacity=capacity,
            requested_at_unix=requested_at,
        )
        if lease is not None:
            return lease
        time.sleep(poll_seconds)


def shared_agent_worker_status(
    root: Path | None = None,
    capacity: int = SHARED_AGENT_WORKER_CAPACITY,
) -> dict[str, object]:
    """Inspect the aggregate campaign worker pool without disturbing leases."""

    selected_root = (root or shared_agent_worker_root()).resolve()
    _ensure_scheduler(selected_root, capacity)
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
                payload = json.loads(handle.read() or "{}")
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                holder = payload
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
    occupied_count = sum(bool(slot["occupied"]) for slot in slots)
    return {
        "schema_version": "1.0",
        "root": str(selected_root),
        "capacity": capacity,
        "occupied": occupied_count,
        "available": capacity - occupied_count,
        "synchronization_barrier": False,
        "slots": slots,
    }

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from experiments.c0c3_factorial.agent_scheduler import (
    WorkerQueueCancelled,
    acquire_agent_worker_slot,
    release_agent_worker_slot,
    shared_agent_worker_status,
    try_acquire_agent_worker_slot,
)


def test_shared_agent_worker_pool_is_cross_campaign_and_bounded(
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent-workers"
    first = try_acquire_agent_worker_slot(
        worker_id="campaign-a:run-1:1",
        metadata={"campaign": "a"},
        root=root,
        capacity=2,
    )
    second = try_acquire_agent_worker_slot(
        worker_id="campaign-b:run-1:1",
        metadata={"campaign": "b"},
        root=root,
        capacity=2,
    )
    assert first is not None
    assert second is not None
    assert (
        try_acquire_agent_worker_slot(
            worker_id="campaign-c:run-1:1",
            root=root,
            capacity=2,
        )
        is None
    )
    status = shared_agent_worker_status(root, capacity=2)
    assert status["capacity"] == 2
    assert status["occupied"] == 2
    assert status["available"] == 0
    assert status["synchronization_barrier"] is False
    assert {
        slot["holder"]["campaign"]
        for slot in status["slots"]
        if slot["occupied"]
    } == {"a", "b"}

    release_agent_worker_slot(first)
    replacement = try_acquire_agent_worker_slot(
        worker_id="campaign-c:run-1:1",
        root=root,
        capacity=2,
    )
    assert replacement is not None
    release_agent_worker_slot(replacement)
    release_agent_worker_slot(second)
    assert shared_agent_worker_status(root, capacity=2)["occupied"] == 0


def test_waiting_worker_can_be_cancelled_before_attempt_start(
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent-workers"
    pause = tmp_path / "pause-request.json"
    pause.write_text("{}\n", encoding="utf-8")

    with pytest.raises(WorkerQueueCancelled):
        acquire_agent_worker_slot(
            worker_id="campaign:run:1",
            root=root,
            capacity=1,
            cancel_path=pause,
            poll_seconds=0.001,
        )

    assert shared_agent_worker_status(root, capacity=1)["occupied"] == 0


def test_scheduler_rejects_capacity_drift(tmp_path: Path) -> None:
    root = tmp_path / "agent-workers"
    assert shared_agent_worker_status(root, capacity=2)["capacity"] == 2
    with pytest.raises(RuntimeError, match="capacity mismatch"):
        shared_agent_worker_status(root, capacity=3)


def test_worker_slot_releases_when_holder_process_exits(tmp_path: Path) -> None:
    root = tmp_path / "agent-workers"
    code = """
import sys
import time
from pathlib import Path
from experiments.c0c3_factorial.agent_scheduler import acquire_agent_worker_slot

lease = acquire_agent_worker_slot(worker_id="child", root=Path(sys.argv[1]), capacity=1)
print("locked", flush=True)
time.sleep(60)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(root)],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"
        assert shared_agent_worker_status(root, capacity=1)["occupied"] == 1
    finally:
        child.terminate()
        child.wait(timeout=10)
    assert shared_agent_worker_status(root, capacity=1)["occupied"] == 0

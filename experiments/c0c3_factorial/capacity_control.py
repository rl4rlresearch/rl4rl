"""Live, cooperative per-campaign concurrency controls.

The limits in this file are operational ceilings rather than scientific
budgets.  Lowering one never interrupts an active subject call or evaluator;
new work waits until the campaign is naturally below the requested ceiling.
Writing these controls never changes a campaign's running/paused lifecycle.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from .state import atomic_json, utc_now

CAMPAIGN_CAPACITY_CONTROL = Path("capacity-control.json")
CAMPAIGN_CAPACITY_LOCK = Path(".capacity-control.lock")
CAMPAIGN_EVALUATOR_ROOT_ENV = "RL4RL_CAMPAIGN_EVALUATOR_ROOT"
MAX_SUBJECT_WORKERS = 30
# Compatibility value for older scientific runtimes that import this name.
# The live dashboard no longer uses it as a hard ceiling; the host-local
# evaluator limit is the authoritative upper bound for new controller writes.
MAX_LOCAL_EVALUATORS = 16


@dataclass(frozen=True)
class CampaignCapacityLimits:
    subject_workers: int
    local_evaluators: int
    updated_at: str | None = None


def _bounded(value: object, *, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def load_campaign_capacity(
    campaign: str | Path,
    *,
    default_subject_workers: int,
    default_local_evaluators: int,
) -> CampaignCapacityLimits:
    """Read current limits, falling back to the campaign's launch defaults."""

    root = Path(campaign).resolve()
    worker_default = (
        MAX_SUBJECT_WORKERS
        if default_subject_workers == 0
        else default_subject_workers
    )
    path = root / CAMPAIGN_CAPACITY_CONTROL
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        payload = {}
    return CampaignCapacityLimits(
        subject_workers=_bounded(
            payload.get("subject_workers", worker_default),
            name="subject_workers",
            maximum=MAX_SUBJECT_WORKERS,
        ),
        local_evaluators=_bounded(
            payload.get("local_evaluators", default_local_evaluators),
            name="local_evaluators",
        ),
        updated_at=(
            str(payload["updated_at"]) if payload.get("updated_at") else None
        ),
    )


@contextmanager
def _capacity_lock(campaign: Path) -> Iterator[None]:
    with (campaign / CAMPAIGN_CAPACITY_LOCK).open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def set_campaign_capacity(
    campaign: str | Path,
    *,
    subject_workers: int,
    local_evaluators: int,
) -> CampaignCapacityLimits:
    """Atomically set live ceilings without pausing or restarting a campaign."""

    root = Path(campaign).resolve()
    if not (root / "campaign.json").is_file():
        raise ValueError(f"campaign does not exist: {root}")
    value = CampaignCapacityLimits(
        subject_workers=_bounded(
            subject_workers,
            name="subject_workers",
            maximum=MAX_SUBJECT_WORKERS,
        ),
        local_evaluators=_bounded(
            local_evaluators,
            name="local_evaluators",
        ),
        updated_at=utc_now(),
    )
    with _capacity_lock(root):
        atomic_json(
            root / CAMPAIGN_CAPACITY_CONTROL,
            {
                "schema_version": "1.0",
                **asdict(value),
                "semantics": "live_future_dispatches_only_no_lifecycle_change",
            },
        )
    return value


def campaign_evaluator_slot_root(campaign: str | Path) -> Path:
    """Return a stable host-local pool unique to one campaign directory."""

    root = Path(campaign).resolve()
    configured = os.environ.get(CAMPAIGN_EVALUATOR_ROOT_ENV)
    if configured:
        base = Path(configured).expanduser().resolve()
    else:
        stable_tmp = Path("/private/tmp")
        temporary = stable_tmp if stable_tmp.is_dir() else Path(tempfile.gettempdir())
        base = temporary / f"rl4rl-c0c3-campaign-evaluators-{os.getuid()}-v1"
    identifier = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    return base / identifier


@dataclass
class CampaignEvaluatorLease:
    handle: TextIO
    path: Path
    index: int


def try_acquire_campaign_evaluator(
    campaign: str | Path,
    *,
    capacity: int,
    opportunity_root: Path,
) -> CampaignEvaluatorLease | None:
    """Try one campaign-specific evaluator slot without blocking."""

    capacity = _bounded(capacity, name="local_evaluators")
    root = campaign_evaluator_slot_root(campaign)
    root.mkdir(parents=True, exist_ok=True)
    first = (
        int.from_bytes(
            hashlib.sha256(str(opportunity_root).encode()).digest()[:8], "little"
        )
        % capacity
    )
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
                "scope": "campaign_dynamic",
                "campaign": str(Path(campaign).resolve()),
                "opportunity_root": str(opportunity_root),
                "acquired_at": utc_now(),
            },
            handle,
            sort_keys=True,
        )
        handle.flush()
        return CampaignEvaluatorLease(handle=handle, path=path, index=index)
    return None


def release_campaign_evaluator(lease: CampaignEvaluatorLease | None) -> None:
    if lease is None or lease.handle.closed:
        return
    fcntl.flock(lease.handle.fileno(), fcntl.LOCK_UN)
    lease.handle.close()


def campaign_evaluator_status(
    campaign: str | Path, *, hard_capacity: int | None = None
) -> dict[str, object]:
    """Inspect active campaign-specific evaluator leases without disturbing them."""

    root = campaign_evaluator_slot_root(campaign)
    root.mkdir(parents=True, exist_ok=True)
    if hard_capacity is None:
        existing_indices = []
        for path in root.glob("slot-*.lock"):
            try:
                existing_indices.append(int(path.stem.removeprefix("slot-")))
            except ValueError:
                continue
        hard_capacity = max(existing_indices, default=0) + 1
    hard_capacity = _bounded(hard_capacity, name="local_evaluators")
    active = 0
    for index in range(hard_capacity):
        handle = (root / f"slot-{index:02d}.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            active += 1
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
    return {"root": str(root), "active": active}

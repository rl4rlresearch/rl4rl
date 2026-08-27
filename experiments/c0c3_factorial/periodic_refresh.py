"""Incumbent-preserving, history-free search epochs for semantic v4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .state import SearchController, append_jsonl, atomic_json, utc_now

PRESERVE = "preserve"
REFRESH_INCUMBENT = "refresh_incumbent_at_phase_start"
POLICIES = frozenset({PRESERVE, REFRESH_INCUMBENT})
MEMORY_STATE = Path("subject-memory.json")


def phase_search_seed(base_seed: int, opportunity: int) -> int:
    payload = f"semantic-periodic-refresh\0{base_seed}\0{opportunity}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def memory_state(run_dir: Path, *, base_seed: int) -> dict[str, Any]:
    path = run_dir / MEMORY_STATE
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value
    return {
        "schema_version": "1.0",
        "history_start_opportunity": 1,
        "search_seed": base_seed,
        "phase": 1,
    }


def apply_periodic_refresh(
    run_dir: Path,
    *,
    controller: SearchController,
    base_seed: int,
    opportunity: int,
) -> dict[str, Any]:
    """Apply one idempotent refresh immediately before a phase begins."""

    path = run_dir / MEMORY_STATE
    current = memory_state(run_dir, base_seed=base_seed)
    if int(current.get("history_start_opportunity", 1)) == opportunity:
        return current
    seed = phase_search_seed(base_seed, opportunity)
    archive_path = run_dir / "developmental-archive.json"
    if archive_path.is_file():
        archive = json.loads(archive_path.read_text(encoding="utf-8"))
        append_jsonl(
            run_dir / "developmental-archive-resets.jsonl",
            {
                "schema_version": "1.0",
                "event": "developmental_archive_reset",
                "timestamp": utc_now(),
                "opportunity": opportunity,
                "archive": archive,
            },
        )
    controller.refresh_from_incumbent(
        opportunity=opportunity,
        search_seed=seed,
        reason="configured five-proposal full search refresh",
    )
    atomic_json(
        archive_path,
        {
            "schema_version": "1.0",
            "items": [],
            "history_start_opportunity": opportunity,
        },
    )
    value = {
        "schema_version": "1.0",
        "policy": REFRESH_INCUMBENT,
        "history_start_opportunity": opportunity,
        "search_seed": seed,
        "phase": 1 + (opportunity - 1) // 5,
        "incumbent_id": controller.state.incumbent_id,
        "updated_at": utc_now(),
    }
    atomic_json(path, value)
    return value

"""Frozen campaign-level execution order."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .spec import FactorialSpec
from .state import SearchController


@dataclass(frozen=True)
class NextRun:
    run_id: str
    condition: str
    block: int
    order: int
    opportunity: int


def _schedule(campaign: Path) -> list[dict[str, object]]:
    value = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("campaign schedule must be a list")
    return sorted(value, key=lambda row: (int(row["block"]), int(row["order"])))


def next_run(campaign_dir: str | Path, spec: FactorialSpec) -> NextRun | None:
    """Choose the least-advanced run, then frozen block/order.

    This produces one opportunity per run per round. It balances provider drift,
    thermal/load effects, and operator timing across conditions more closely than
    completing all 100 opportunities of one condition before starting the next.
    """

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
    return NextRun(
        run_id=controller.state.run_id,
        condition=str(assignment["condition"]),
        block=block,
        order=order,
        opportunity=controller.state.next_opportunity,
    )

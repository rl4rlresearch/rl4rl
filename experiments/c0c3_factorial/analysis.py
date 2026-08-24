"""Transparent 2x2 point estimators for precomputed per-run outcomes."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .spec import Condition


@dataclass(frozen=True)
class RunOutcome:
    block: str
    condition: Condition
    qualified_mechanism_clusters: int
    run_id: str


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot estimate a contrast from an empty cell")
    return sum(values) / len(values)


def estimate(outcomes: list[RunOutcome]) -> dict[str, object]:
    run_ids = [row.run_id for row in outcomes]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("factorial outcomes contain duplicate run IDs")
    block_cells = [(row.block, row.condition) for row in outcomes]
    if len(block_cells) != len(set(block_cells)):
        raise ValueError("factorial outcomes contain duplicate block-condition cells")
    cells = {
        condition: [
            float(row.qualified_mechanism_clusters)
            for row in outcomes
            if row.condition is condition
        ]
        for condition in Condition
    }
    counts = {condition: len(values) for condition, values in cells.items()}
    if len(set(counts.values())) != 1 or 0 in counts.values():
        raise ValueError(f"factorial cells must be non-empty and balanced: {counts}")
    means = {condition: _mean(values) for condition, values in cells.items()}
    memory = ((means[Condition.C2] + means[Condition.C3]) / 2) - (
        (means[Condition.C0] + means[Condition.C1]) / 2
    )
    transition = ((means[Condition.C1] + means[Condition.C3]) / 2) - (
        (means[Condition.C0] + means[Condition.C2]) / 2
    )
    interaction = (means[Condition.C3] - means[Condition.C2]) - (
        means[Condition.C1] - means[Condition.C0]
    )
    blocks: dict[str, set[Condition]] = {}
    for outcome in outcomes:
        blocks.setdefault(outcome.block, set()).add(outcome.condition)
    incomplete = {
        block: sorted(
            condition.value for condition in Condition if condition not in cells
        )
        for block, cells in blocks.items()
        if cells != set(Condition)
    }
    if incomplete:
        raise ValueError(f"incomplete randomized blocks: {incomplete}")
    block_contrasts = []
    for block in sorted(blocks):
        values = {
            outcome.condition: float(outcome.qualified_mechanism_clusters)
            for outcome in outcomes
            if outcome.block == block
        }
        block_contrasts.append(
            {
                "block": block,
                "portfolio_memory_effect": (
                    (values[Condition.C2] + values[Condition.C3]) / 2
                    - (values[Condition.C0] + values[Condition.C1]) / 2
                ),
                "assumption_changing_effect": (
                    (values[Condition.C1] + values[Condition.C3]) / 2
                    - (values[Condition.C0] + values[Condition.C2]) / 2
                ),
                "interaction": (values[Condition.C3] - values[Condition.C2])
                - (values[Condition.C1] - values[Condition.C0]),
            }
        )
    return {
        "estimand": "distinct Layer-B-qualified mechanism clusters per run",
        "cell_means": {key.value: value for key, value in means.items()},
        "cell_counts": {key.value: value for key, value in counts.items()},
        "portfolio_memory_main_effect": memory,
        "assumption_changing_main_effect": transition,
        "interaction": interaction,
        "block_contrasts": block_contrasts,
    }


def read_outcomes(path: str | Path) -> list[RunOutcome]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        RunOutcome(
            block=row["block"],
            condition=Condition(row["condition"]),
            qualified_mechanism_clusters=int(row["qualified_mechanism_clusters"]),
            run_id=row["run_id"],
        )
        for row in rows
    ]


def write_estimate(path: str | Path, value: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

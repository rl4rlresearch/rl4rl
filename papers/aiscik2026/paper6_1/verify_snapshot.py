#!/usr/bin/env python3
"""Verify the frozen Paper 6.1 analytic snapshot without live campaign state."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"

HORIZONS = {"addition": 80, "fashion": 200, "nanogpt": 40}
RUNS = {"addition": 20, "fashion": 20, "nanogpt": 12}
PROPOSALS = {task: HORIZONS[task] * RUNS[task] for task in HORIZONS}
CHECKPOINT_PAIRS = {"addition": 80, "fashion": 200, "nanogpt": 24}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DERIVED / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def close(left: float, right: float, tolerance: float = 1e-10) -> None:
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"numeric mismatch: {left} != {right}")


def block_mean(rows: list[dict[str, str]], field: str) -> float:
    by_block: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if math.isfinite(value):
            by_block[int(row["block"])].append(value)
    return mean([mean(values) for values in by_block.values()])


def main() -> None:
    proposals = read_csv("proposal_records.csv")
    pairs = read_csv("checkpoint_pairs.csv")
    effects = read_csv("checkpoint_effects.csv")
    cycles = read_csv("cycle_gain_pairs.csv")
    cycle_effects = read_csv("cycle_gain_effects.csv")
    integrity = json.loads((DERIVED / "integrity.json").read_text(encoding="utf-8"))

    assert len(proposals) == sum(PROPOSALS.values()) == 6080
    assert len(pairs) == sum(CHECKPOINT_PAIRS.values()) == 304
    assert integrity["checkpoint_opportunities"] == 608
    assert integrity["checkpoint_messages_available"] == 605

    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_run: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in proposals:
        task = row["task"]
        by_task[task].append(row)
        by_run[(task, row["run_id"])].append(row)
        opportunity = int(row["opportunity"])
        treated = int(row["treated"])
        intervention = int(row["intervention"])
        expected = int(bool(treated) and opportunity % 10 == 0)
        assert intervention == expected

    for task in HORIZONS:
        assert len(by_task[task]) == PROPOSALS[task]
        task_runs = [rows for (name, _), rows in by_run.items() if name == task]
        assert len(task_runs) == RUNS[task]
        for rows in task_runs:
            assert {int(row["opportunity"]) for row in rows} == set(range(1, HORIZONS[task] + 1))

    for task, expected in CHECKPOINT_PAIRS.items():
        assert sum(row["task"] == task for row in pairs) == expected

    # Recompute every all-memory point estimate in the primary effect table
    # from the frozen matched-pair rows.
    pair_metrics = {
        key[4:]
        for key in pairs[0]
        if key.startswith("did_")
    }
    for row in effects:
        if row["memory"] != "all":
            continue
        task = row["task"]
        metric = row["metric"]
        if metric not in pair_metrics:
            continue
        subset = [pair for pair in pairs if pair["task"] == task]
        close(block_mean(subset, f"did_{metric}"), float(row["did_effect"]))

    # Recompute immediate, follow-up, and total cycle effects.
    for row in cycle_effects:
        task = row["task"]
        metric = row["metric"]
        subset = [cycle for cycle in cycles if cycle["task"] == task]
        close(block_mean(subset, f"difference_{metric}"), float(row["paired_difference"]))

    source_available = sum(int(row["source_available"]) for row in proposals)
    assert source_available == 5802
    print("Paper 6.1 frozen snapshot verified: 52 runs, 6,080 proposals, 304 matched checkpoints.")


if __name__ == "__main__":
    main()

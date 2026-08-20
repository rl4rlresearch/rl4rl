#!/usr/bin/env python3
"""Run a bounded batch of independently closed one-union macro-attempts."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


WORKSPACE = Path(__file__).resolve().parents[1]
RUN_DIR = WORKSPACE.parent
MODEL = WORKSPACE / "src" / "model.py"
PAIR_RE = re.compile(r"groups (\d+) and (\d+)")


def read_state() -> dict:
    return json.loads((RUN_DIR / "STATE.json").read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parameter_tying_results() -> list[dict[str, str]]:
    return [
        row
        for row in read_rows(RUN_DIR / "RESULTS.tsv")
        if "Family: parameter tying." in row.get("proposal", "")
    ]


def evidence_label(row: dict[str, str]) -> str:
    return (
        f"{row['attempt_id']} ({row['status']}, "
        f"{row.get('parameters') or 'no parameter count'} parameters, "
        f"{row.get('accuracy') or 'no score'})"
    )


def family_evidence() -> tuple[int, str, str]:
    family = parameter_tying_results()
    accepted = next(row for row in reversed(family) if row["status"] == "keep")
    failed = next(row for row in reversed(family) if row["status"] != "keep")
    return len(family), evidence_label(accepted), evidence_label(failed)


def frontier_accuracy(current: dict) -> float:
    attempt_id = str(current["incumbent"]["attempt_id"])
    if "/" in attempt_id:
        macro, micro = attempt_id.split("/", 1)
        for row in reversed(read_rows(RUN_DIR / "AUTOMATION_RESULTS.tsv")):
            if row["macro_attempt_id"] == macro and row["micro_attempt_id"] == micro:
                return float(row["accuracy"].rstrip("%"))
    else:
        for row in reversed(read_rows(RUN_DIR / "RESULTS.tsv")):
            if row["attempt_id"] == attempt_id:
                return float(row["accuracy"].rstrip("%"))
    raise RuntimeError(f"accuracy not found for incumbent {attempt_id}")


def rejected_pairs() -> set[tuple[int, int]]:
    rejected: set[tuple[int, int]] = set()
    for row in read_rows(RUN_DIR / "AUTOMATION_RESULTS.tsv"):
        if row["family"] != "parameter tying" or row["status"] != "discard":
            continue
        match = PAIR_RE.search(row["description"])
        if match:
            rejected.add(tuple(sorted((int(match.group(1)), int(match.group(2))))))
    return rejected


def candidate_pairs() -> list[tuple[float, int, int, float, float]]:
    checkpoint = torch.load(
        RUN_DIR / "state" / "incumbent.pt", map_location="cpu", weights_only=False
    )["model_state"]
    values = checkpoint["pos_emb.values"]
    mapping = checkpoint["pos_emb.value_indices"].flatten()
    sorted_values, sorted_groups = torch.sort(values)
    excluded = rejected_pairs()
    candidates: list[tuple[float, int, int, float, float]] = []
    for index in range(len(sorted_values) - 1):
        left_group = int(sorted_groups[index])
        right_group = int(sorted_groups[index + 1])
        left_coord = int((mapping == left_group).nonzero()[0])
        right_coord = int((mapping == right_group).nonzero()[0])
        if tuple(sorted((left_coord, right_coord))) in excluded:
            continue
        left_value = float(values[left_group])
        right_value = float(values[right_group])
        candidates.append(
            (
                abs(right_value - left_value),
                left_coord,
                right_coord,
                left_value,
                right_value,
            )
        )
    candidates.sort()
    return candidates


def add_pair(left: int, right: int) -> None:
    source = MODEL.read_text(encoding="utf-8")
    boundary = source.index("\n\n        self.blocks")
    closing = source.rfind(")))", 0, boundary)
    if closing < 0:
        raise RuntimeError("could not locate tied_pairs terminator")
    MODEL.write_text(
        source[:closing]
        + f"),\n                        ({left}, {right})))"
        + source[closing + 3 :],
        encoding="utf-8",
    )


def run_runner(*arguments: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(RUN_DIR / "run_attempt.py"), "--run-dir", str(RUN_DIR), *arguments],
        cwd=WORKSPACE,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"runner exited {completed.returncode}: {' '.join(arguments)}")


def run_one() -> str:
    current = read_state()
    attempt_number = int(current["attempts_used"]) + 1
    parameters = int(current["incumbent"]["parameters"])
    accuracy = frontier_accuracy(current)
    prior, accepted, failed = family_evidence()
    candidates = candidate_pairs()
    if len(candidates) < 2:
        raise RuntimeError("fewer than two eligible positional pairs remain")
    difference, left, right, left_value, right_value = candidates[0]
    alternative = candidates[1][0]
    automation_id = f"position-union-{attempt_number:04d}"
    description = f"Test nearest eligible position groups {left} and {right}"
    proposal = (
        f"Family: parameter tying. Current retained frontier is {parameters:,} parameters at "
        f"{accuracy:.6f}%, a +{accuracy - 99:.6f} percentage-point margin over 99%. There have "
        f"been {prior} prior attempts in this family. The most recent accepted result is "
        f"{accepted}; the most recent failed result is {failed}. Hypothesis: the one-parameter, "
        f"parameterization-preserving union of current positional groups represented by coordinates "
        f"{left} and {right}, with values {left_value:.10f} and {right_value:.10f} (difference "
        f"{difference:.3e}), preserves qualification using a multiplicity-weighted mean and yields "
        f"{parameters - 1:,} parameters. Candidate ordering is ascending current absolute group-value "
        f"difference with previously nonqualifying exact pairs excluded. This is more informative than "
        f"the nearest untested alternative (difference {alternative:.3e}) because it is the eligible "
        f"minimum-distortion merge. The official runner alone accepts or rolls back. Stop after exactly "
        f"one micro-trial or immediately on error. Compute budget: one checkpoint migration, one official "
        f"verification, and two minutes wall-clock."
    )
    print(
        f"attempt {attempt_number}: pair={left},{right} diff={difference:.3e} "
        f"frontier={parameters}/{accuracy:.2f}%",
        flush=True,
    )
    run_runner(
        "automation-start",
        "--automation-id",
        automation_id,
        "--family",
        "parameter tying",
        "--description",
        description,
        "--proposal",
        proposal,
        "--max-micro-trials",
        "1",
    )
    add_pair(left, right)
    started = time.monotonic()
    run_runner(
        "automation-attempt",
        "--description",
        f"Union closest position groups {left} and {right}",
        "--proposal",
        proposal,
    )
    result = read_rows(RUN_DIR / "AUTOMATION_RESULTS.tsv")[-1]
    elapsed = time.monotonic() - started
    status = result["status"]
    if status == "keep":
        outcome = (
            f"1 accepted, 0 discarded, 0 errors; frontier is {result['parameters']} parameters "
            f"at {result['accuracy']}"
        )
    elif status == "discard":
        outcome = (
            f"0 accepted, 1 discarded, 0 errors; score was {result['accuracy']} at "
            f"{result['parameters']} parameters and the prior frontier was restored"
        )
    else:
        outcome = "0 accepted, 0 discarded, 1 error; prior frontier was restored"
    run_runner(
        "automation-end",
        "--summary",
        f"Completed the fixed one-micro-trial boundary in {elapsed:.1f} seconds: {outcome}. "
        f"Tested group representatives {left}/{right} at difference {difference:.3e}; stop reason "
        f"was the declared one-trial cap.",
    )
    print(f"attempt {attempt_number} closed: {status}", flush=True)
    return status


def main() -> int:
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    if batch < 1 or batch > 20:
        raise ValueError("batch size must be between 1 and 20")
    for _ in range(batch):
        current = read_state()
        config = json.loads((RUN_DIR / "RUN_CONFIG.json").read_text(encoding="utf-8"))
        if int(current["attempts_used"]) >= int(config["max_attempts"]):
            print("Configured attempt budget exhausted.", flush=True)
            return 0
        status = run_one()
        if status == "error":
            print("Stopping batch after recorded error for manual log inspection.", flush=True)
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Bounded nearest-value unions over block-0 pre-FF normalization biases."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch


WORKSPACE = Path(__file__).resolve().parents[1]
RUN_DIR = WORKSPACE.parent
MODEL = WORKSPACE / "src" / "model.py"
BASE_PRIOR_FAMILY_ATTEMPTS = 83
BASE_PRIOR_MICRO_TRIALS = 673
EXCLUDED = {(1, 14), (5, 8)}


def state() -> dict:
    return json.loads((RUN_DIR / "STATE.json").read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_family_evidence() -> tuple[str, str]:
    accepted = "attempt-0094/micro-0001 at 1,673 parameters and 99.04%"
    failed = "attempt-0094/micro-0002 (discard at 97.90%)"
    for row in rows(RUN_DIR / "AUTOMATION_RESULTS.tsv"):
        if row["family"] != "parameter tying":
            continue
        label = f"{row['macro_attempt_id']}/{row['micro_attempt_id']}"
        if row["status"] == "keep":
            accepted = f"{label} at {row['parameters']} parameters and {row['accuracy']}"
        elif row["status"] in {"discard", "error"}:
            failed = f"{label} ({row['status']}, {row['accuracy'] or 'no score'})"
    return accepted, failed


def incumbent_accuracy(current: dict) -> float:
    macro, micro = current["incumbent"]["attempt_id"].split("/", 1)
    for row in reversed(rows(RUN_DIR / "AUTOMATION_RESULTS.tsv")):
        if row["macro_attempt_id"] == macro and row["micro_attempt_id"] == micro:
            return float(row["accuracy"].rstrip("%"))
    raise RuntimeError("could not locate incumbent accuracy")


def candidate_pairs() -> list[tuple[float, int, int, float, float]]:
    checkpoint = torch.load(
        RUN_DIR / "state" / "incumbent.pt", map_location="cpu", weights_only=False
    )["model_state"]
    values = checkpoint["blocks.0.ln2.bias_values"]
    mapping = checkpoint["blocks.0.ln2.bias_indices"]
    sorted_values, sorted_groups = torch.sort(values)
    candidates = []
    for index in range(len(sorted_values) - 1):
        left_group = int(sorted_groups[index])
        right_group = int(sorted_groups[index + 1])
        left_coord = int((mapping == left_group).nonzero()[0])
        right_coord = int((mapping == right_group).nonzero()[0])
        if tuple(sorted((left_coord, right_coord))) in EXCLUDED:
            continue
        left = float(values[left_group])
        right = float(values[right_group])
        candidates.append((abs(right - left), left_coord, right_coord, left, right))
    candidates.sort()
    return candidates


def add_pair(left: int, right: int) -> None:
    source = MODEL.read_text(encoding="utf-8")
    start = source.index("BLOCK0_LN2_BIAS_TIED_PAIRS = (")
    boundary = source.index("\n)\n\n\nclass TiedTokenEmbedding", start)
    insertion = f"\n    ({left}, {right}),"
    MODEL.write_text(source[:boundary] + insertion + source[boundary:], encoding="utf-8")


def main() -> int:
    while True:
        current = state()
        automation = current["active_automation"]
        used = int(automation["micro_attempts_used"])
        cap = int(automation["max_micro_trials"])
        if used >= cap:
            print(f"Reached declared micro-trial cap ({cap}).", flush=True)
            return 0
        candidates = candidate_pairs()
        if not candidates:
            print("No block-0 pre-FF bias pair remains.", flush=True)
            return 0
        difference, left, right, left_value, right_value = candidates[0]
        alternative = candidates[1][0] if len(candidates) > 1 else float("nan")
        accepted, failed = latest_family_evidence()
        parameters = int(current["incumbent"]["parameters"])
        accuracy = incumbent_accuracy(current)
        add_pair(left, right)
        description = f"Union closest block-0 pre-FF norm biases {left} and {right}"
        proposal = (
            f"Family: parameter tying. Current retained frontier is {parameters:,} parameters "
            f"at {accuracy:.6f}%, a +{accuracy - 99:.6f} percentage-point margin over 99%. "
            f"There have been {BASE_PRIOR_FAMILY_ATTEMPTS} prior macro-attempts in this family "
            f"and {BASE_PRIOR_MICRO_TRIALS + used} prior micro-trials. The most recent accepted "
            f"result is {accepted}; the most recent failed result is {failed}. This is a "
            f"parameterization-preserving compression of block 0's pre-feed-forward normalization "
            f"biases: union coordinates {left} and {right}, values {left_value:.10f} and "
            f"{right_value:.10f} (difference {difference:.3e}), using a weighted mean and integer "
            f"reconstruction. Hypothesis: additive centering has local redundancy distinct from "
            f"the measured scale boundary, preserving qualification at {parameters - 1:,} "
            f"parameters. It is more informative than the next bias union (difference "
            f"{alternative:.3e}) because it is the minimum-distortion isolated bias test; scales "
            f"remain fixed. Official acceptance and rollback are unchanged."
        )
        completed = subprocess.run(
            [sys.executable, str(RUN_DIR / "run_attempt.py"), "--run-dir", str(RUN_DIR),
             "automation-attempt", "--description", description, "--proposal", proposal],
            cwd=WORKSPACE, capture_output=True, text=True)
        if completed.returncode != 0:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            return completed.returncode
        result = rows(RUN_DIR / "AUTOMATION_RESULTS.tsv")[-1]
        if result["status"] in {"discard", "error"}:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            print(f"Recorded {result['status']} closes block-0 pre-FF bias path.", flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())

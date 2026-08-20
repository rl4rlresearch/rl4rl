#!/usr/bin/env python3
"""Bounded nearest-value unions over block-0 pre-attention norm scales."""

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
BASE_PRIOR_FAMILY_ATTEMPTS = 38
BASE_PRIOR_MICRO_TRIALS = 448


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def state() -> dict:
    return json.loads((RUN_DIR / "STATE.json").read_text(encoding="utf-8"))


def evidence() -> tuple[str, str]:
    accepted = "attempt-0095/micro-0001 at 1,672 parameters and 99.01%"
    failed = "attempt-0095/micro-0002 (discard at 98.78%)"
    for row in rows(RUN_DIR / "AUTOMATION_RESULTS.tsv"):
        if row["family"] != "parameter tying":
            continue
        label = f"{row['macro_attempt_id']}/{row['micro_attempt_id']}"
        if row["status"] == "keep":
            accepted = f"{label} at {row['parameters']} parameters and {row['accuracy']}"
        elif row["status"] in {"discard", "error"}:
            failed = f"{label} ({row['status']}, {row['accuracy'] or 'no score'})"
    return accepted, failed


def accuracy(current: dict) -> float:
    macro, micro = current["incumbent"]["attempt_id"].split("/", 1)
    for row in reversed(rows(RUN_DIR / "AUTOMATION_RESULTS.tsv")):
        if row["macro_attempt_id"] == macro and row["micro_attempt_id"] == micro:
            return float(row["accuracy"].rstrip("%"))
    raise RuntimeError("could not locate incumbent accuracy")


def candidates() -> list[tuple[float, int, int, float, float]]:
    checkpoint = torch.load(
        RUN_DIR / "state" / "incumbent.pt", map_location="cpu", weights_only=False
    )["model_state"]
    if "blocks.0.ln1.weight_group_indices" in checkpoint:
        values = checkpoint["blocks.0.ln1.weight_values"]
        coords = checkpoint["blocks.0.ln1.weight_coordinates"]
        groups = checkpoint["blocks.0.ln1.weight_group_indices"]
    else:
        values = checkpoint["blocks.0.ln1.weight"]
        coords = torch.arange(values.numel())
        groups = torch.arange(values.numel())
    sorted_values, sorted_groups = torch.sort(values)
    result = []
    for index in range(len(sorted_values) - 1):
        left_group = int(sorted_groups[index])
        right_group = int(sorted_groups[index + 1])
        left = int(coords[(groups == left_group).nonzero()[0]])
        right = int(coords[(groups == right_group).nonzero()[0]])
        left_value = float(values[left_group])
        right_value = float(values[right_group])
        result.append((abs(right_value - left_value), left, right, left_value, right_value))
    return sorted(result)


def add_pair(left: int, right: int) -> None:
    source = MODEL.read_text(encoding="utf-8")
    start = source.index("BLOCK0_LN1_WEIGHT_TIED_PAIRS = (")
    boundary = source.index("\n)\n\nBLOCK0_LN1_BIAS_TIED_PAIRS", start)
    MODEL.write_text(
        source[:boundary] + f"\n    ({left}, {right})," + source[boundary:],
        encoding="utf-8")


def main() -> int:
    while True:
        current = state()
        active = current["active_automation"]
        used = int(active["micro_attempts_used"])
        cap = int(active["max_micro_trials"])
        if used >= cap:
            print(f"Reached declared micro-trial cap ({cap}).", flush=True)
            return 0
        options = candidates()
        if not options:
            print("No block-0 pre-attention scale pair remains.", flush=True)
            return 0
        difference, left, right, left_value, right_value = options[0]
        alternative = options[1][0] if len(options) > 1 else float("nan")
        accepted, failed = evidence()
        parameters = int(current["incumbent"]["parameters"])
        retained_accuracy = accuracy(current)
        add_pair(left, right)
        proposal = (
            f"Family: parameter tying. Current retained frontier is {parameters:,} parameters "
            f"at {retained_accuracy:.6f}%, a +{retained_accuracy - 99:.6f} percentage-point "
            f"margin over 99%. There have been {BASE_PRIOR_FAMILY_ATTEMPTS} prior macro-attempts "
            f"and {BASE_PRIOR_MICRO_TRIALS + used} prior micro-trials in this family. The most "
            f"recent accepted result is {accepted}; the most recent failed result is {failed}. "
            f"This is a parameterization-preserving compression of block 0's pre-attention "
            f"normalization scales: union coordinates {left} and {right}, values "
            f"{left_value:.10f} and {right_value:.10f} (difference {difference:.3e}), using a "
            f"weighted mean and integer reconstruction. Hypothesis: attention-input scaling has "
            f"local redundancy distinct from pre-FF normalization, preserving qualification at "
            f"{parameters - 1:,} parameters. It is more informative than the next scale union "
            f"(difference {alternative:.3e}) because it is the minimum-distortion isolated test; "
            f"the two fixed-zero biases remain fixed. Official acceptance and rollback are unchanged."
        )
        completed = subprocess.run(
            [sys.executable, str(RUN_DIR / "run_attempt.py"), "--run-dir", str(RUN_DIR),
             "automation-attempt", "--description",
             f"Union closest block-0 pre-attention norm scales {left} and {right}",
             "--proposal", proposal], cwd=WORKSPACE, capture_output=True, text=True)
        if completed.returncode != 0:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            return completed.returncode
        result = rows(RUN_DIR / "AUTOMATION_RESULTS.tsv")[-1]
        if result["status"] in {"discard", "error"}:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            print(f"Recorded {result['status']} closes block-0 pre-attention scale path.", flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())

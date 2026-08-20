#!/usr/bin/env python3
"""Run all bounded nearest-value unions over final LayerNorm scales."""

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
BASE_PRIOR_FAMILY_ATTEMPTS = 80
BASE_PRIOR_MICRO_TRIALS = 664
EXCLUDED = {(0, 2)}


def state() -> dict:
    return json.loads((RUN_DIR / "STATE.json").read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_family_evidence() -> tuple[str, str]:
    accepted = "attempt-0092/micro-0003 at 1,683 parameters and 99.00%"
    failed = "attempt-0092/micro-0004 (discard at 98.98%)"
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
    if "ln_f.values" in checkpoint:
        values = checkpoint["ln_f.values"]
        mapping = checkpoint["ln_f.value_indices"]
    else:
        values = checkpoint["ln_f.weight"]
        mapping = torch.arange(values.numel())
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
    start = source.index("LN_F_TIED_PAIRS = (")
    boundary = source.index("\n)\n\nBLOCK0_LN2_WEIGHT_TIED_PAIRS", start)
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
            print("No final-normalization scale pair remains.", flush=True)
            return 0
        difference, left, right, left_value, right_value = candidates[0]
        alternative = candidates[1][0] if len(candidates) > 1 else float("nan")
        accepted, failed = latest_family_evidence()
        parameters = int(current["incumbent"]["parameters"])
        accuracy = incumbent_accuracy(current)
        add_pair(left, right)
        description = f"Union closest final LayerNorm scale groups {left} and {right}"
        proposal = (
            f"Family: parameter tying. Current retained frontier is {parameters:,} parameters "
            f"at {accuracy:.6f}%, a +{accuracy - 99:.6f} percentage-point margin over 99%. "
            f"There have been {BASE_PRIOR_FAMILY_ATTEMPTS} prior macro-attempts in this family "
            f"and {BASE_PRIOR_MICRO_TRIALS + used} prior micro-trials. The most recent accepted "
            f"result is {accepted}; the most recent failed result is {failed}. This is a "
            f"parameterization-preserving compression of the final normalization scales: union "
            f"coordinates {left} and {right}, values {left_value:.10f} and {right_value:.10f} "
            f"(difference {difference:.3e}), using a weighted mean and integer reconstruction. "
            f"Hypothesis: the final affine rescaling contains a near-duplicate coordinate pair "
            f"independent of matrix and positional boundaries, preserving qualification at "
            f"{parameters - 1:,} parameters. It is more informative than the next scale union "
            f"(difference {alternative:.3e}) because it is the minimum-distortion test in the "
            f"complete 16-scale search space. Official acceptance and rollback are unchanged."
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
            print(f"Recorded {result['status']} closes final-norm scale path.", flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
